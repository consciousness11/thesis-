"""Gymnasium environment wrapping the WorkflowSim TCP bridge.

Each Gymnasium episode corresponds to one CloudSim simulation on the Java
side.  This env is the *server*: it listens, accepts one Java client (the
JVM running ``PPOTrainerExample`` or ``PPOSchedulingAlgorithmExample``), and
exchanges frames per the protocol documented in
``cozy-fluttering-leaf.md`` and ``protocol.py``.

Observation is a fixed-shape dict so MaskablePPO's MultiInputPolicy can
consume it without dynamic shapes.  Padding masks expose the variable
workflow size and idle-VM set.
"""

from __future__ import annotations

import math
import socket
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from protocol import FrameError, recv_frame, send_frame


MAX_TASKS = 128          # padded upper bound for DAG node count
MAX_VMS = 64             # action space size; assert in Java that VM count <= 64
NODE_FEAT_DIM = 4        # log_length, depth, parent_count, child_count
VM_FEAT_DIM = 4          # mips_norm, elec_cost, pending_finish_norm, is_idle
TASK_FEAT_DIM = 5        # log_length, depth, parent_count, child_count, task_idx_norm


class WorkflowSimEnv(gym.Env):
    """Synchronous Gymnasium env driven by a Java TCP client."""

    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 7777):
        super().__init__()
        self._host = host
        self._port = port
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None

        self._dag_nodes: list[dict] = []
        self._dag_edges: list[tuple[int, int]] = []
        self._task_id_to_idx: dict[int, int] = {}
        self._vm_id_to_idx: dict[int, int] = {}
        self._vm_idx_to_id: dict[int, int] = {}
        self._vm_static_features: np.ndarray = np.zeros(
            (MAX_VMS, VM_FEAT_DIM), dtype=np.float32
        )
        self._current_mask: np.ndarray = np.zeros(MAX_VMS, dtype=np.int8)
        self._adjacency: np.ndarray = np.zeros(
            (MAX_TASKS, MAX_TASKS), dtype=np.float32
        )
        self._node_features: np.ndarray = np.zeros(
            (MAX_TASKS, NODE_FEAT_DIM), dtype=np.float32
        )
        self._node_mask: np.ndarray = np.zeros(MAX_TASKS, dtype=np.int8)
        self._last_obs: Optional[dict[str, np.ndarray]] = None
        self._episode_summary: Optional[dict[str, Any]] = None
        self._hello_seen = False

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-10.0, high=10.0,
                    shape=(MAX_TASKS, NODE_FEAT_DIM), dtype=np.float32,
                ),
                "node_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_TASKS,), dtype=np.int8,
                ),
                "adjacency": spaces.Box(
                    low=0, high=1,
                    shape=(MAX_TASKS, MAX_TASKS), dtype=np.float32,
                ),
                "vm_features": spaces.Box(
                    low=-10.0, high=10.0,
                    shape=(MAX_VMS, VM_FEAT_DIM), dtype=np.float32,
                ),
                "task_features": spaces.Box(
                    low=-10.0, high=10.0,
                    shape=(TASK_FEAT_DIM,), dtype=np.float32,
                ),
                "vm_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_VMS,), dtype=np.int8,
                ),
            }
        )
        self.action_space = spaces.Discrete(MAX_VMS)

    # ------------------------------------------------------------------ networking

    def _open_server(self) -> None:
        if self._server is not None:
            return
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self._host, self._port))
        self._server.listen(1)
        print(f"[server] listening on {self._host}:{self._port}")

    def _accept_client(self) -> None:
        assert self._server is not None
        self._client, addr = self._server.accept()
        self._client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server] client connected from {addr}")
        self._hello_seen = False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass
            self._client = None
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None

    # ------------------------------------------------------------------ frame I/O

    def _next_frame(self) -> dict:
        assert self._client is not None
        frame = recv_frame(self._client)
        if frame is None:
            raise FrameError("client closed connection mid-episode")
        return frame

    def _send(self, frame: dict) -> None:
        assert self._client is not None
        send_frame(self._client, frame)

    # ------------------------------------------------------------------ DAG cache

    def _ingest_episode_start(self, frame: dict) -> None:
        dag = frame.get("dag", {})
        nodes = dag.get("nodes", [])
        edges = dag.get("edges", [])
        vms = frame.get("vms", [])

        self._dag_nodes = nodes
        self._dag_edges = [tuple(e) for e in edges]

        self._task_id_to_idx = {n["id"]: i for i, n in enumerate(nodes[:MAX_TASKS])}
        self._node_features.fill(0.0)
        self._node_mask.fill(0)
        self._adjacency.fill(0.0)
        for i, n in enumerate(nodes[:MAX_TASKS]):
            length = float(n.get("length", 0))
            self._node_features[i] = np.array(
                [
                    math.log1p(max(0.0, length)),
                    float(n.get("depth", 0)),
                    0.0,  # parent_count, filled below
                    0.0,  # child_count, filled below
                ],
                dtype=np.float32,
            )
            self._node_mask[i] = 1
        for parent_id, child_id in self._dag_edges:
            pi = self._task_id_to_idx.get(parent_id)
            ci = self._task_id_to_idx.get(child_id)
            if pi is None or ci is None:
                continue
            self._adjacency[ci, pi] = 1.0  # incoming edge for GAT
            self._node_features[pi, 3] += 1.0  # child_count of parent
            self._node_features[ci, 2] += 1.0  # parent_count of child

        self._vm_id_to_idx = {}
        self._vm_idx_to_id = {}
        self._vm_static_features.fill(0.0)
        for i, vm in enumerate(vms[:MAX_VMS]):
            vid = int(vm["id"])
            self._vm_id_to_idx[vid] = i
            self._vm_idx_to_id[i] = vid
            mips = float(vm.get("mips", 0)) * float(vm.get("pes", 1))
            self._vm_static_features[i] = np.array(
                [mips / 1000.0, float(vm.get("elec_cost", 1.0)), 0.0, 0.0],
                dtype=np.float32,
            )

    # ------------------------------------------------------------------ obs builder

    def _build_obs(self, observe: dict) -> dict[str, np.ndarray]:
        idle_ids = list(observe.get("idle_vm_ids", []))
        vm_features = self._vm_static_features.copy()
        self._current_mask = np.zeros(MAX_VMS, dtype=np.int8)
        for vid in idle_ids:
            idx = self._vm_id_to_idx.get(int(vid))
            if idx is None:
                continue
            vm_features[idx, 3] = 1.0
            self._current_mask[idx] = 1
        # pending finish per VM (Java sends [mips, elec, pending] per idle VM)
        for arr, vid in zip(observe.get("vm_features", []), idle_ids):
            idx = self._vm_id_to_idx.get(int(vid))
            if idx is None or len(arr) < 3:
                continue
            vm_features[idx, 2] = float(arr[2]) / 100.0  # pending normalised

        tf = observe.get("task_features", {})
        task_idx = self._task_id_to_idx.get(
            int(observe.get("task_id", -1)), -1
        )
        task_features = np.array(
            [
                math.log1p(max(0.0, float(tf.get("length", 0)))),
                float(tf.get("depth", 0)),
                float(tf.get("parent_count", 0)),
                float(tf.get("child_count", 0)),
                float(task_idx) / MAX_TASKS,
            ],
            dtype=np.float32,
        )

        return {
            "node_features": self._node_features.copy(),
            "node_mask": self._node_mask.copy(),
            "adjacency": self._adjacency.copy(),
            "vm_features": vm_features,
            "task_features": task_features,
            "vm_mask": self._current_mask.copy(),
        }

    # ------------------------------------------------------------------ gym api

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if self._server is None:
            self._open_server()
        if self._client is None:
            self._accept_client()

        # Wait for either HELLO (first run) then EPISODE_START, or directly
        # an EPISODE_START on subsequent resets.
        while True:
            frame = self._next_frame()
            ftype = frame.get("type")
            if ftype == "HELLO":
                self._hello_seen = True
                continue
            if ftype == "EPISODE_START":
                self._ingest_episode_start(frame)
                break
            raise FrameError(f"unexpected frame before EPISODE_START: {ftype}")

        observe = self._next_frame()
        if observe.get("type") != "OBSERVE":
            raise FrameError(f"expected OBSERVE first, got {observe.get('type')}")
        obs = self._build_obs(observe)
        self._last_obs = obs
        self._episode_summary = None
        return obs, {}

    def action_masks(self) -> np.ndarray:
        """Hook used by sb3_contrib.MaskablePPO."""
        return self._current_mask.astype(bool)

    def step(self, action: int):
        if not self._current_mask[int(action)]:
            # MaskablePPO should prevent this; if it slips through pick
            # any valid VM so the simulation doesn't deadlock.
            valid = np.flatnonzero(self._current_mask)
            action = int(valid[0]) if len(valid) else int(action)

        vm_id = self._vm_idx_to_id.get(int(action))
        if vm_id is None:
            # No mapping (should not happen post-masking); send any idle id
            valid_ids = [
                vid for vid, idx in self._vm_id_to_idx.items()
                if self._current_mask[idx]
            ]
            vm_id = valid_ids[0] if valid_ids else 0

        self._send({"type": "ACTION", "vm_id": vm_id})

        step_result = self._next_frame()
        if step_result.get("type") != "STEP_RESULT":
            raise FrameError(
                f"expected STEP_RESULT after ACTION, got {step_result.get('type')}"
            )
        reward = float(step_result.get("reward", 0.0))

        next_frame = self._next_frame()
        ntype = next_frame.get("type")
        if ntype == "OBSERVE":
            obs = self._build_obs(next_frame)
            self._last_obs = obs
            return obs, reward, False, False, {}
        if ntype == "EPISODE_END":
            terminal = float(next_frame.get("terminal_reward", 0.0))
            self._episode_summary = {
                "makespan": float(next_frame.get("makespan", 0.0)),
                "energy": float(next_frame.get("energy", 0.0)),
                "jobs_completed": int(next_frame.get("jobs_completed", 0)),
            }
            return self._last_obs, reward + terminal, True, False, self._episode_summary
        raise FrameError(f"unexpected frame after STEP_RESULT: {ntype}")
