"""DQN vs PPO regression gate.

Parses the stdout of the two example runs and prints a side-by-side delta
table.  Exit code is 0 iff PPO is within 2x of DQN on both makespan and
energy for every workflow (criterion 2 in the plan).

Typical usage::

    java -cp "build;lib/*" \
        org.workflowsim.examples.scheduling.DQNSchedulingAlgorithmExample \
        > /tmp/dqn.log

    # in two terminals:
    python python/infer.py --port 7778 --checkpoint python/runs/ppo_smoke/ppo_smoke.zip
    java -Dppo.port=7778 -cp "build;lib/*" \
        org.workflowsim.examples.scheduling.PPOSchedulingAlgorithmExample > /tmp/ppo.log

    python python/compare.py /tmp/dqn.log /tmp/ppo.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


WORKFLOWS = ("Montage", "CyberShake", "Epigenomics", "LIGO")


def parse_dqn(text: str) -> dict[str, dict[str, float]]:
    """Extract per-workflow makespan / energy from the DQN summary table."""
    out: dict[str, dict[str, float]] = {}
    in_summary = False
    for line in text.splitlines():
        if "MOPWSDRL summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        m = re.match(r"\s*(\S+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
        if not m:
            continue
        wf = m.group(1)
        if wf in WORKFLOWS:
            out[wf] = {"makespan": float(m.group(2)), "energy": float(m.group(3))}
    return out


def parse_ppo(text: str) -> dict[str, dict[str, float]]:
    """Extract per-workflow makespan / energy from PPO inference lines."""
    out: dict[str, dict[str, float]] = {}
    pat = re.compile(
        r"workflow=(\S+)\s+makespan=([\d.]+)\s+energy=([\d.]+)"
    )
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        wf = m.group(1)
        if wf in WORKFLOWS:
            out[wf] = {"makespan": float(m.group(2)), "energy": float(m.group(3))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dqn_log", type=Path)
    parser.add_argument("ppo_log", type=Path)
    args = parser.parse_args()

    dqn = parse_dqn(args.dqn_log.read_text(encoding="utf-8"))
    ppo = parse_ppo(args.ppo_log.read_text(encoding="utf-8"))

    print(f"{'Workflow':<14}{'DQN_make':>12}{'PPO_make':>12}{'Δ%':>8}   "
          f"{'DQN_energy':>14}{'PPO_energy':>14}{'Δ%':>8}")
    overall_ok = True
    for wf in WORKFLOWS:
        dm = dqn.get(wf, {}).get("makespan")
        de = dqn.get(wf, {}).get("energy")
        pm = ppo.get(wf, {}).get("makespan")
        pe = ppo.get(wf, {}).get("energy")
        if dm is None or pm is None:
            print(f"{wf:<14}{'?':>12}{'?':>12}{'?':>8}   "
                  f"{'?':>14}{'?':>14}{'?':>8}")
            overall_ok = False
            continue
        dmake = (pm - dm) / dm * 100.0
        denergy = (pe - de) / de * 100.0
        flag_make = pm <= 2.0 * dm
        flag_energy = pe <= 2.0 * de
        if not (flag_make and flag_energy):
            overall_ok = False
        print(
            f"{wf:<14}{dm:>12.2f}{pm:>12.2f}{dmake:>+7.1f}%   "
            f"{de:>14.2f}{pe:>14.2f}{denergy:>+7.1f}%"
        )

    if overall_ok:
        print("\nPASS: PPO within 2x of DQN on every workflow.")
        sys.exit(0)
    print("\nFAIL: PPO exceeded 2x of DQN on at least one workflow.")
    sys.exit(1)


if __name__ == "__main__":
    main()
