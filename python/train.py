"""Train the PPO + GAT scheduler against the WorkflowSim Java client.

Run me first, in one terminal::

    python python/train.py --port 7777 --total-timesteps 3000

Then, in a second terminal::

    java -Dppo.port=7777 -Dppo.episodes=10 -cp "build;lib/*" \
        org.workflowsim.examples.scheduling.PPOTrainerExample

The Python process owns the TCP server and the PPO loop; Java is the
client that drives CloudSim episodes.  When training finishes the policy
is saved to ``python/runs/ppo_smoke/ppo_smoke.zip``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python python/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--total-timesteps", type=int, default=3000)
    parser.add_argument("--output-dir", default="python/runs/ppo_smoke")
    parser.add_argument("--checkpoint-name", default="ppo_smoke.zip")
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "sb3_contrib not installed. Run `pip install -r python/requirements.txt`"
        ) from exc

    from workflowsim_env import WorkflowSimEnv
    from policy import WorkflowFeaturesExtractor
    from protocol import FrameError

    os.makedirs(args.output_dir, exist_ok=True)
    env = WorkflowSimEnv(host=args.host, port=args.port)

    policy_kwargs = dict(
        features_extractor_class=WorkflowFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log=str(Path(args.output_dir) / "tb"),
    )

    print(f"[train] starting MaskablePPO learn for {args.total_timesteps} timesteps")
    try:
        model.learn(total_timesteps=args.total_timesteps)
    except KeyboardInterrupt:
        print("[train] interrupted; saving partial model")
    except FrameError as exc:
        print(f"[train] Java client disconnected ({exc}); treating as clean stop")
    finally:
        out_path = Path(args.output_dir) / args.checkpoint_name
        model.save(out_path)
        print(f"[train] saved checkpoint -> {out_path}")
        env.close()


if __name__ == "__main__":
    main()
