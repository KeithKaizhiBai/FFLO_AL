from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = Path(
    "hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/active_learning_flowchart"
)


def git_commit_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_stage(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    number: str,
    title: str,
    lines: list[str],
    face: str,
    edge: str,
    accent: str,
) -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.014",
        linewidth=1.35,
        facecolor=face,
        edgecolor=edge,
        zorder=2,
    )
    ax.add_patch(box)
    badge = FancyBboxPatch(
        (x + 0.012, y + h - 0.054),
        0.037,
        0.037,
        boxstyle="round,pad=0.004,rounding_size=0.006",
        linewidth=0,
        facecolor=accent,
        edgecolor=accent,
        zorder=3,
    )
    ax.add_patch(badge)
    ax.text(x + 0.0305, y + h - 0.0355, number, ha="center", va="center", fontsize=10.2, color="white", weight="bold")
    ax.text(x + 0.061, y + h - 0.0355, title, ha="left", va="center", fontsize=11.3, color="#102a43", weight="bold")
    for i, line in enumerate(lines):
        ax.text(x + 0.028, y + h - 0.086 - i * 0.038, f"- {line}", ha="left", va="top", fontsize=8.25, color="#243b53")


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    rad: float = 0.0,
    lw: float = 1.85,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def add_label(ax, xy: tuple[float, float], text: str, color: str = "#52616b") -> None:
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=8.3, color=color)


def build_flowchart(output_dir: Path) -> dict:
    setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit_or_none()

    fig = plt.figure(figsize=(14.4, 7.2), constrained_layout=False)
    ax = fig.add_axes([0.035, 0.055, 0.93, 0.9])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "Active-Learning Workflow for Exact BdG Phase-Boundary Refinement",
        ha="center",
        va="center",
        fontsize=15.2,
        color="#102a43",
        weight="bold",
    )
    ax.text(
        0.5,
        0.918,
        "ML schedules exact BdG calls; exact BdG remains the source of physical labels.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#52616b",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.055, 0.19),
            0.89,
            0.62,
            boxstyle="round,pad=0.014,rounding_size=0.024",
            linewidth=0.9,
            facecolor="#f8fafc",
            edgecolor="#d8e0ea",
            zorder=0,
        )
    )

    blue = ("#eff6ff", "#93c5fd", "#2563eb")
    amber = ("#fffbeb", "#f3c969", "#b7791f")
    violet = ("#f5f3ff", "#b8a5f4", "#7c3aed")
    green = ("#ecfdf5", "#86d39b", "#2f855a")

    add_stage(
        ax,
        (0.08, 0.58),
        (0.245, 0.19),
        "1",
        "Learning Dataset",
        [
            r"$21528\rightarrow24083$ exact points",
            r"$x=(k_B T/t,\;J_A/t)$",
            r"$y_{\rm reg},\ y_{\rm phase}$, q/Delta status",
        ],
        *blue,
    )
    add_stage(
        ax,
        (0.3775, 0.58),
        (0.245, 0.19),
        "2",
        "ML Training",
        [
            r"regression: $\Delta_{\rm opt},q_{\rm opt},\eta,I_c^\pm$",
            r"classifier: normal / uniform_SC / FFLO",
            r"uncertainty: $\sigma_y,\ U_{\rm cls},\ H$",
        ],
        *amber,
    )
    add_stage(
        ax,
        (0.675, 0.58),
        (0.245, 0.19),
        "3",
        "Acquisition",
        [
            r"dense $x=(k_B T/t,\;J_A/t)$ candidates",
            r"score $S(x)$: boundary + q/Delta + eta risks",
            r"existing-point exclusion + diversity",
        ],
        *amber,
    )
    add_stage(
        ax,
        (0.675, 0.32),
        (0.245, 0.19),
        "4",
        "HPC Exact BdG Oracle",
        [
            r"selected batch on HPC",
            r"minimize $F(\Delta,q)$",
            r"$I_c^\pm,\ \eta,\ y_{\rm phase}$, exact status",
        ],
        *violet,
    )
    add_stage(
        ax,
        (0.3775, 0.32),
        (0.245, 0.19),
        "5",
        "Quality Gate",
        [
            r"trusted $\rightarrow$ training eligible",
            r"boundary-band $\rightarrow$ metadata preserved",
            r"unresolved $\rightarrow$ rerun list",
        ],
        *green,
    )
    add_stage(
        ax,
        (0.08, 0.32),
        (0.245, 0.19),
        "6",
        "Append + Diagnostics",
        [
            r"write $\mathrm{dataset\_iter}_{n+1}$",
            r"diagnostics + boundary extraction",
            r"append exact labels, then retrain",
        ],
        *green,
    )

    add_arrow(ax, (0.325, 0.675), (0.3775, 0.675), "#2563eb")
    add_arrow(ax, (0.6225, 0.675), (0.675, 0.675), "#b7791f")
    add_arrow(ax, (0.7975, 0.58), (0.7975, 0.51), "#7c3aed")
    add_arrow(ax, (0.675, 0.415), (0.6225, 0.415), "#2f855a")
    add_arrow(ax, (0.3775, 0.415), (0.325, 0.415), "#2f855a")
    add_arrow(ax, (0.2025, 0.51), (0.2025, 0.58), "#334e68", rad=-0.08)

    add_label(ax, (0.352, 0.704), "train view")
    add_label(ax, (0.650, 0.704), "score")
    add_label(ax, (0.840, 0.545), "selected exact calls")
    add_label(ax, (0.650, 0.392), "merge + filter")
    add_label(ax, (0.352, 0.392), "append")
    add_label(ax, (0.162, 0.545), "retrain")

    ax.text(0.085, 0.245, "Key rule", ha="left", va="center", fontsize=10.0, weight="bold", color="#102a43")
    ax.text(
        0.17,
        0.245,
        r"ML chooses expensive exact points. Physical labels and boundary evidence come from exact BdG outputs.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#334e68",
    )
    ax.text(0.085, 0.215, "Current scope", ha="left", va="center", fontsize=10.0, weight="bold", color="#102a43")
    ax.text(
        0.17,
        0.215,
        "normal/SC and uniform/FFLO use active-learning exact data; topological boundaries remain reference-only.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#334e68",
    )

    png = output_dir / "active_learning_flowchart_v2.png"
    pdf = output_dir / "active_learning_flowchart_v2.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "output_png": str(png),
        "output_pdf": str(pdf),
        "warm_start_exact_points": 21528,
        "current_exact_points": 24083,
        "current_dataset": "dataset_iter042",
        "script": str(Path("scripts/plot_active_learning_flowchart_v2.py")),
        "commit": commit,
        "design_notes": [
            "The workflow is shown as a six-stage clockwise loop.",
            "ML training and acquisition are separate scheduler-side stages.",
            "The HPC exact BdG oracle is separated from the quality gate and append/retrain logic.",
            "Trusted, boundary-band, and rerun-required exact outputs are explicitly distinguished.",
        ],
    }
    (output_dir / "active_learning_flowchart_v2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the revised active-learning workflow.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_flowchart(args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
