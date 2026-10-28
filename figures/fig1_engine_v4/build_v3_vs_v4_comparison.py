#!/usr/bin/env python3
"""Build the human-review comparison for frozen v3 versus Candidate-A v4."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "fig1_engine_v3" / "fig1_engine_v3.png"
V4 = HERE / "fig1_engine_v4.png"


def central_crop(image: Image.Image) -> Image.Image:
    scale_x = image.width / 1600.0
    scale_y = image.height / 1080.0
    box = (325, 150, 1250, 840)
    return image.crop(tuple(round(value * scale) for value, scale in zip(box, (scale_x, scale_y, scale_x, scale_y))))


def main() -> int:
    v3 = Image.open(V3).convert("RGB")
    v4 = Image.open(V4).convert("RGB")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), dpi=180)
    for axis, image, title in (
        (axes[0, 0], v3, "Frozen engine-v3"),
        (axes[0, 1], v4, "Candidate-A engine-v4"),
        (axes[1, 0], central_crop(v3), "v3 central physics"),
        (axes[1, 1], central_crop(v4), "v4 central physics"),
    ):
        axis.imshow(image)
        axis.set_title(title, fontsize=11, weight="semibold")
        axis.axis("off")
    fig.suptitle("Figure 1: frozen v3 versus Candidate-A landscape integration", fontsize=15, weight="semibold")
    fig.text(0.5, 0.018,
             "Only the thermodynamic landscape and its immediate local labels/axes change; topology and peripheral cards remain frozen.",
             ha="center", fontsize=8.5, color="#46515E")
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.96), h_pad=1.0, w_pad=0.8)
    fig.savefig(HERE / "fig1_engine_v3_vs_v4.pdf", format="pdf", dpi=300, facecolor="white")
    fig.savefig(HERE / "fig1_engine_v3_vs_v4.png", format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
