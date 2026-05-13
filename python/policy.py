"""Custom SB3 feature extractor combining GAT(DAG), VM MLP, and the current
task descriptor.

MaskablePPO's MultiInputPolicy plugs into the dict observation produced by
:class:`workflowsim_env.WorkflowSimEnv` and feeds it through this extractor
before the standard MLP policy/value heads.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gat_encoder import GATEncoder


class WorkflowFeaturesExtractor(BaseFeaturesExtractor):
    """Produces a fixed-size feature vector from the dict observation."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        node_dim = observation_space["node_features"].shape[1]
        vm_dim = observation_space["vm_features"].shape[1]
        task_dim = observation_space["task_features"].shape[0]
        self.max_vms = observation_space["vm_features"].shape[0]

        self.gat = GATEncoder(in_dim=node_dim, hidden_dim=64, heads=4)

        self.vm_mlp = nn.Sequential(
            nn.Linear(vm_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.task_mlp = nn.Sequential(
            nn.Linear(task_dim, 16),
            nn.ReLU(),
        )
        combined_in = self.gat.out_dim + 32 + 16
        self.head = nn.Sequential(
            nn.Linear(combined_in, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        node_features = obs["node_features"].float()
        adjacency = obs["adjacency"].float()
        node_mask = obs["node_mask"].float()
        vm_features = obs["vm_features"].float()
        task_features = obs["task_features"].float()
        vm_mask = obs["vm_mask"].float()

        graph_embed = self.gat(node_features, adjacency, node_mask)        # (B, 64)

        vm_h = self.vm_mlp(vm_features)                                    # (B, MAX_VMS, 32)
        m = vm_mask.unsqueeze(-1)
        vm_pooled = (vm_h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)    # (B, 32)

        task_embed = self.task_mlp(task_features)                          # (B, 16)

        combined = torch.cat([graph_embed, vm_pooled, task_embed], dim=-1)
        return self.head(combined)
