from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_phase.config import ActiveLearningConfig


DEFAULT_OUTPUT_DIR = Path(
    "hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/figures/ml_training_architecture"
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


def add_box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    lines: list[str],
    face: str,
    edge: str,
    title_size: float = 10.3,
    line_size: float = 8.15,
    line_gap: float = 0.035,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=1.25,
            facecolor=face,
            edgecolor=edge,
            zorder=2,
        )
    )
    ax.text(x + 0.014, y + h - 0.028, title, ha="left", va="top", fontsize=title_size, weight="bold", color="#102a43")
    for i, line in enumerate(lines):
        ax.text(x + 0.017, y + h - 0.067 - i * line_gap, line, ha="left", va="top", fontsize=line_size, color="#243b53")


def add_neural_network(
    ax,
    xy: tuple[float, float],
    output_dim: int,
    face: str,
    edge: str,
    line: str,
) -> None:
    x, y = xy
    layer_x = [x, x + 0.070, x + 0.140, x + 0.218]
    layers = [
        [y + 0.052, y + 0.018],
        [y + 0.078, y + 0.054, y + 0.030, y + 0.006],
        [y + 0.078, y + 0.054, y + 0.030, y + 0.006],
        [y + 0.078 - 0.020 * i for i in range(output_dim)],
    ]
    radius = 0.0068

    for src_i in range(len(layers) - 1):
        for y0 in layers[src_i]:
            for y1 in layers[src_i + 1]:
                ax.plot(
                    [layer_x[src_i] + radius, layer_x[src_i + 1] - radius],
                    [y0, y1],
                    color=line,
                    linewidth=0.45,
                    alpha=0.32,
                    zorder=3,
                )

    for lx, ys in zip(layer_x, layers):
        for yy in ys:
            ax.add_patch(
                Circle(
                    (lx, yy),
                    radius,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.0,
                    zorder=5,
                )
            )

    labels = [
        r"$2$",
        r"$64$",
        r"$64$",
        rf"${output_dim}$",
    ]
    for lx, label in zip(layer_x, labels):
        ax.text(lx, y - 0.010, label, ha="center", va="top", fontsize=8.0, color="#102a43")


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    rad: float = 0.0,
    lw: float = 1.65,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=4,
        )
    )


def add_network_box(
    ax,
    xy: tuple[float, float],
    title: str,
    output_dim: int,
    loss_formula: str,
    loss_note: str,
    face: str,
    edge: str,
) -> None:
    x, y = xy
    add_box(ax, xy, (0.330, 0.250), title, [], face, edge, title_size=10.4)
    ax.text(x + 0.020, y + 0.198, r"$x_s=(x-\mu_x)/\sigma_x$", ha="left", va="top", fontsize=8.4, color="#243b53")
    add_neural_network(
        ax,
        (x + 0.052, y + 0.095),
        output_dim,
        "#ffffff",
        edge,
        "#64748b",
    )
    ax.text(x + 0.020, y + 0.058, r"fully connected; ReLU after hidden layers", ha="left", va="top", fontsize=7.3, color="#52616b")
    ax.text(x + 0.020, y + 0.034, loss_formula, ha="left", va="top", fontsize=7.8, color="#243b53")
    ax.text(x + 0.020, y + 0.008, loss_note, ha="left", va="top", fontsize=7.8, color="#243b53")


def build_figure(output_dir: Path) -> dict:
    setup_matplotlib()
    cfg = ActiveLearningConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit_or_none()

    fig = plt.figure(figsize=(15.6, 9.2), constrained_layout=False)
    ax = fig.add_axes([0.035, 0.055, 0.93, 0.90])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.5, 0.965, "ML Training Architecture in the Active-Learning Loop", ha="center", va="center", fontsize=15.2, weight="bold", color="#102a43")
    ax.text(
        0.5,
        0.928,
        "The networks learn a surrogate and uncertainty map for candidate selection; exact BdG remains the source of physical labels.",
        ha="center",
        va="center",
        fontsize=9.3,
        color="#52616b",
    )

    add_box(
        ax,
        (0.055, 0.620),
        (0.245, 0.210),
        "Exact training data",
        [
            r"$x=(k_B T/t,\;J_A/t)$",
            r"$y_{\rm reg}=(\Delta_{\rm opt},q_{\rm opt},\eta,I_c^+,I_c^-)$",
            r"$y_{\rm phase}\in\{\mathrm{normal},\mathrm{uniform\_SC},\mathrm{FFLO}\}$",
            r"$n_{\rm current}=24083$ exact points",
        ],
        "#eff6ff",
        "#93c5fd",
        title_size=10.5,
        line_size=8.2,
        line_gap=0.039,
    )
    add_box(
        ax,
        (0.055, 0.340),
        (0.245, 0.195),
        "Split and scaling",
        [
            rf"stratified validation fraction $={cfg.val_fraction:g}$",
            r"$x_s=(x-\mu_x)/\sigma_x$",
            r"$y_{{\rm reg},s}=(y_{\rm reg}-\mu_y)/\sigma_y$",
            r"class labels are mapped to integer indices",
        ],
        "#eff6ff",
        "#93c5fd",
        title_size=10.5,
        line_size=8.2,
        line_gap=0.039,
    )

    add_network_box(
        ax,
        (0.350, 0.615),
        rf"Regression Ensemble ($M={cfg.n_ensemble}$)",
        5,
        r"$\mathcal{L}_{\rm reg}=N^{-1}\sum_i\|\hat y_{i,s}-y_{i,s}\|_2^2$",
        "loss: MSELoss on standardized targets",
        "#fffaf0",
        "#f3c969",
    )
    add_network_box(
        ax,
        (0.350, 0.320),
        rf"Phase Classifier Ensemble ($M={cfg.n_ensemble}$)",
        3,
        r"$\mathcal{L}_{\rm cls}=-N^{-1}\sum_i\log p_{i,y_i}$",
        "loss: CrossEntropyLoss on phase labels",
        "#f5f3ff",
        "#b8a5f4",
    )

    add_box(
        ax,
        (0.745, 0.615),
        (0.215, 0.210),
        "Regression products",
        [
            r"$\bar y=\langle \hat y^{(m)}\rangle_m$",
            r"$\sigma_y=\mathrm{std}_m(\hat y^{(m)})$",
            r"$\Delta_{\rm opt},q_{\rm opt},\eta,I_c^+,I_c^-$",
            "mean + uncertainty feed acquisition",
        ],
        "#fffaf0",
        "#f3c969",
        title_size=10.5,
        line_size=8.2,
        line_gap=0.039,
    )
    add_box(
        ax,
        (0.745, 0.320),
        (0.215, 0.210),
        "Classifier products",
        [
            r"$p(c|x)=\mathrm{softmax}(z_c)$",
            r"$U_{\rm cls}=1-\max_c p(c|x)$",
            r"$H=-\sum_c p_c\log p_c/\log C$",
            "phase probabilities feed acquisition",
        ],
        "#f5f3ff",
        "#b8a5f4",
        title_size=10.5,
        line_size=8.2,
        line_gap=0.039,
    )

    add_box(
        ax,
        (0.055, 0.170),
        (0.895, 0.105),
        "Training parameters",
        [
            f"hidden_dim={cfg.hidden_dim}   batch_size={cfg.batch_size}   reg_epochs={cfg.reg_epochs}   "
            f"cls_epochs={cfg.cls_epochs}   lr={cfg.lr:g}   weight_decay={cfg.weight_decay:g}",
            f"optimizer=Adam   seed={cfg.seed}   device=cuda if available, else cpu",
        ],
        "#f8fafc",
        "#cbd5e1",
        title_size=10.0,
        line_size=8.2,
        line_gap=0.031,
    )
    add_box(
        ax,
        (0.055, 0.025),
        (0.895, 0.110),
        "Validation metrics reported downstream",
        [
            r"$\mathrm{RMSE}(\Delta_{\rm opt}),\ \mathrm{RMSE}(q_{\rm opt}),\ \mathrm{RMSE}(\eta),\ \mathrm{RMSE}(I_c^+),\ \mathrm{RMSE}(I_c^-)$",
            r"$\mathrm{phase\ accuracy}$ and boundary diagnostics; these are diagnostics, not a summed training loss",
        ],
        "#f8fafc",
        "#cbd5e1",
        title_size=10.0,
        line_size=8.4,
        line_gap=0.027,
    )

    add_arrow(ax, (0.177, 0.620), (0.177, 0.535), "#2563eb")
    add_arrow(ax, (0.300, 0.438), (0.350, 0.740), "#b7791f", rad=-0.18)
    add_arrow(ax, (0.300, 0.438), (0.350, 0.445), "#7c3aed")
    add_arrow(ax, (0.680, 0.740), (0.745, 0.740), "#b7791f")
    add_arrow(ax, (0.680, 0.445), (0.745, 0.445), "#7c3aed")
    ax.text(0.712, 0.765, "ensemble mean/std", ha="center", va="center", fontsize=7.8, color="#52616b")
    ax.text(0.712, 0.470, "softmax + entropy", ha="center", va="center", fontsize=7.8, color="#52616b")

    png = output_dir / "ml_training_architecture.png"
    pdf = output_dir / "ml_training_architecture.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "output_png": str(png),
        "output_pdf": str(pdf),
        "script": str(Path("scripts/plot_ml_training_architecture.py")),
        "commit": commit,
        "model": {
            "input_dim": 2,
            "hidden_dim": cfg.hidden_dim,
            "regression_output_dim": 5,
            "classification_output_dim": 3,
            "n_ensemble": cfg.n_ensemble,
            "architecture": "Linear(2,64)-ReLU-Linear(64,64)-ReLU-Linear(64,out_dim)",
            "figure_rendering": "Representative neuron circles with fully connected lines; hidden layers are labeled as 64 neurons.",
        },
        "training_parameters": {
            "batch_size": cfg.batch_size,
            "reg_epochs": cfg.reg_epochs,
            "cls_epochs": cfg.cls_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "val_fraction": cfg.val_fraction,
            "seed": cfg.seed,
            "optimizer": "Adam",
            "regression_loss": "MSELoss on standardized regression targets",
            "classification_loss": "CrossEntropyLoss on phase labels",
        },
    }
    (output_dir / "ml_training_architecture_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the ML training architecture used by active learning.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = build_figure(args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
