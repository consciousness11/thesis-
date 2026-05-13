"""Inference-mode counterpart to train.py.

Loads a saved PPO checkpoint and serves actions to a Java client without any
gradient updates.  Pair with the Java side::

    python python/infer.py --port 7778 --checkpoint python/runs/ppo_smoke/ppo_smoke.zip
    java -Dppo.port=7778 -cp "build;lib/*" \
        org.workflowsim.examples.scheduling.PPOSchedulingAlgorithmExample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7778)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "sb3_contrib not installed. Run `pip install -r python/requirements.txt`"
        ) from exc

    from workflowsim_env import WorkflowSimEnv

    env = WorkflowSimEnv(host=args.host, port=args.port)
    model = MaskablePPO.load(args.checkpoint, env=env)
    print(f"[infer] loaded checkpoint {args.checkpoint}")

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_reward = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(
                    obs, deterministic=True, action_masks=env.action_masks()
                )
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
            print(f"[infer] episode {ep + 1}: reward={ep_reward:.4f} summary={info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
