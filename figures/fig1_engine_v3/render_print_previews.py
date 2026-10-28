#!/usr/bin/env python3
"""Render 300-dpi color and grayscale print previews from the normalized PDF."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
PDFTOPPM = Path("/Users/baikaizhi/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm")


def main() -> int:
    output_stem = HERE / "fig1_engine_v3"
    subprocess.run(
        [str(PDFTOPPM), "-png", "-r", "300", "-singlefile", str(HERE / "fig1_engine_v3.pdf"), str(output_stem)],
        check=True,
    )
    png = HERE / "fig1_engine_v3.png"
    with Image.open(png).convert("RGB") as image:
        ImageOps.grayscale(image).save(HERE / "fig1_engine_v3_grayscale.png")
        size = list(image.size)
    payload = {"schema_version": 1, "status": "pass", "dpi": 300, "pixel_size": size,
               "color_preview": png.name, "grayscale_preview": "fig1_engine_v3_grayscale.png"}
    (HERE / "print_preview_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
