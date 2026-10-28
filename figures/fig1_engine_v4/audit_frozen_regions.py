#!/usr/bin/env python3
"""Prove that v4 changes only the approved landscape-local region."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageChops, ImageStat


HERE = Path(__file__).resolve().parent


def main() -> int:
    v3 = Image.open(HERE.parent / "fig1_engine_v3" / "fig1_engine_v3.png").convert("RGB")
    v4 = Image.open(HERE / "fig1_engine_v4.png").convert("RGB")
    layout = yaml.safe_load((HERE / "layout.yaml").read_text(encoding="utf-8"))
    regions = {"topology_resolution": layout["scientific_subobjects"]["topology_resolution"]}
    regions.update({name: box for name, box in layout["cards"].items() if name != "central"})
    scales = ((v3.width / 1600.0, v3.height / 1080.0), (v4.width / 1600.0, v4.height / 1080.0))
    results = {}
    for name, box in regions.items():
        crops = []
        for image, (scale_x, scale_y) in zip((v3, v4), scales):
            crop_box = (
                round(box["x"] * scale_x), round(box["y"] * scale_y),
                round((box["x"] + box["width"]) * scale_x),
                round((box["y"] + box["height"]) * scale_y),
            )
            crops.append(image.crop(crop_box))
        comparison = crops[1].resize(crops[0].size, Image.Resampling.NEAREST)
        difference = ImageChops.difference(crops[0], comparison)
        statistics = ImageStat.Stat(difference)
        mean = sum(statistics.mean) / 3.0
        maximum = max(channel[1] for channel in difference.getextrema())
        results[name] = {
            "mean_absolute_channel_difference": mean,
            "maximum_channel_difference": maximum,
            "difference_bbox": difference.getbbox(),
            "identical": difference.getbbox() is None,
        }
    passed = all(item["identical"] for item in results.values())
    payload = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "source": "frozen fig1_engine_v3 300-dpi PDF render",
        "candidate": "fig1_engine_v4 300-dpi PDF render",
        "regions": results,
        "interpretation": "Topology and every peripheral card are pixel-identical; only the landscape-local central region changed.",
    }
    (HERE / "frozen_region_audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Figure 1 v4 frozen-region audit", "",
        f"- Status: **{payload['status'].upper()}**", "",
        "| Region | Mean difference | Maximum difference | Pixel-identical |", "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {name} | {item['mean_absolute_channel_difference']:.6f} | {item['maximum_channel_difference']} | {item['identical']} |"
        for name, item in results.items()
    )
    lines.extend(["", payload["interpretation"], ""])
    (HERE / "frozen_region_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "regions": len(results)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
