"""Bar-chart comparison of HEFT, DQN, and PPO+GAT over the four 100-task
scientific workflows.

Parses three log files in ``results/``:

    results/heft.log   <- HEFTSchedulingAlgorithmExample stdout
    results/dqn.log    <- DQNSchedulingAlgorithmExample stdout
    results/ppo.log    <- PPOTrainerExample stdout (inference mode)

Renders two side-by-side bar charts (makespan, energy) to
``results/comparison.png``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORKFLOWS = ("Montage", "CyberShake", "Epigenomics", "LIGO")

# Three line shapes we accept:
#   [Montage] makespan=125.10  energy=367970.20 W*s  jobs=101         (HEFT)
#   [ep 01/01 Montage     ] makespan=125.10 energy=367970.20 jobs=101 (PPO trainer)
#   Montage              125.10    367970.20      101                  (DQN/HEFT summary)
LABELLED = re.compile(
    r"\b(Montage|CyberShake|Epigenomics|LIGO)\b\s*\]?\s*"
    r"makespan=([\d.]+)\s+energy=([\d.]+)"
)
SUMMARY = re.compile(
    r"^\s*(Montage|CyberShake|Epigenomics|LIGO)\s+"
    r"([\d.]+)\s+([\d.]+)\s+\d+\s*$"
)


def parse_log(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LABELLED.search(line) or SUMMARY.match(line)
        if not m:
            continue
        wf = m.group(1)
        out.setdefault(wf, (float(m.group(2)), float(m.group(3))))
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    results = repo / "results"
    logs = {
        "HEFT":     parse_log(results / "heft.log"),
        "DQN":      parse_log(results / "dqn.log"),
        "Rainbow":  parse_log(results / "rainbow.log"),
        "PPO+GAT":  parse_log(results / "ppo.log"),
    }

    algs = ("HEFT", "DQN", "Rainbow", "PPO+GAT")

    header = f"{'workflow':<14}"
    for alg in algs:
        header += f"{alg + '_mk':>12}"
    for alg in algs:
        header += f"{alg + '_E':>16}"
    print(header)
    for wf in WORKFLOWS:
        cells = [f"{wf:<14}"]
        for alg in algs:
            mk, _ = logs[alg].get(wf, (float("nan"), float("nan")))
            cells.append(f"{mk:>12.2f}")
        for alg in algs:
            _, en = logs[alg].get(wf, (float("nan"), float("nan")))
            cells.append(f"{en:>16.2f}")
        print("".join(cells))

    # ----- plot ----------------------------------------------------------------
    x = np.arange(len(WORKFLOWS))
    width = 0.2
    colors = {"HEFT": "#4C72B0", "DQN": "#DD8452", "Rainbow": "#C44E52", "PPO+GAT": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, metric, title, ylabel in [
        (axes[0], 0, "Makespan", "Makespan (simulated seconds)"),
        (axes[1], 1, "Energy",   "Energy (W·s)"),
    ]:
        for i, alg in enumerate(algs):
            vals = [logs[alg].get(wf, (float("nan"), float("nan")))[metric]
                    for wf in WORKFLOWS]
            offset = (i - (len(algs) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=alg, color=colors[alg])
            for rect, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.annotate(
                    f"{v:,.0f}" if metric == 0 else f"{v:,.0f}",
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, rotation=0,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(WORKFLOWS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if metric == 1:
            ax.set_yscale("log")
            ax.set_ylabel(ylabel + " (log scale)")
        ax.legend(loc="upper left")

    fig.suptitle("HEFT vs DQN vs Rainbow-lite vs PPO+GAT  •  4 workflows × ~1000 tasks  •  20 heterogeneous VMs (mips ∈ {500,1000,1500,2000})  •  PPO sees first 256 of ~1000 DAG nodes",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = results / "comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\n[plot] saved -> {out}")


if __name__ == "__main__":
    main()
