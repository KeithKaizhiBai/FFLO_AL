#!/usr/bin/env python3
"""Rebuild and validate the complete Figure 2 human-review package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: list[str], env: dict[str, str]) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def main() -> int:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": "/tmp/al_fflo_mpl",
                "XDG_CACHE_HOME": "/tmp/al_fflo_cache", "PYTHONPATH": str(ROOT)})
    python = sys.executable
    run([python, str(HERE / "prepare_inputs.py")], env)
    run([python, str(HERE / "build_fig2.py")], env)
    first_png = sha256(HERE / "fig2_strict_data.png")
    run([python, str(HERE / "build_fig2.py")], env)
    second_png = sha256(HERE / "fig2_strict_data.png")
    deterministic = first_png == second_png
    (HERE / "fig2_render_determinism.json").write_text(json.dumps({
        "schema_version": 1, "artifact": "fig2_strict_data.png", "first_sha256": first_png,
        "second_sha256": second_png, "deterministic": deterministic}, indent=2) + "\n", encoding="utf-8")
    if not deterministic:
        raise RuntimeError("Figure 2 PNG rendering is not deterministic")
    run([python, "-m", "figures.figure_engine.validators.figure_lint", "--contract",
         str(HERE / "fig2_panel_contract.yaml"), "--svg", "fig2_strict_data.svg", "--geometry",
         "geometry_manifest.json", "--pdf", "fig2_strict_data.pdf", "--png", "fig2_strict_data.png",
         "--output-dir", str(HERE), "--report-prefix", "figure_validation_report"], env)
    run([python, str(HERE / "validate_fig2_science.py")], env)
    print(json.dumps({"status": "READY", "gate": "GATE_FIG2_HUMAN_SCIENCE_REVIEW",
                      "deterministic_png": deterministic}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
