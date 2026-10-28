from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stagev_v3 import STAGEV3_RUN_ID, StageV3Config


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description="Stage V-v3 package preflight.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    cfg = StageV3Config.from_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pyc = run([
        sys.executable,
        "-m",
        "py_compile",
        "ml_phase/stagev_v3.py",
        "scripts/stagev_v3_select.py",
        "scripts/stagev_v3_update_reward.py",
        "scripts/stagev_v3_smoke.py",
    ])
    status = "pass" if pyc.returncode == 0 and cfg.run_id == STAGEV3_RUN_ID and cfg.mu_min == -0.5 and cfg.mu_max == 1.5 else "fail"
    checks = {
        "status": status,
        "run_id": cfg.run_id,
        "output_root": cfg.output_root,
        "initial_seed_size": cfg.initial_seed_size,
        "micro_batch_size": cfg.micro_batch_size,
        "max_micro_batches": cfg.max_micro_batches,
        "cold_start": True,
        "stageiv_data_used_for_training": False,
        "stagev1_data_used_for_training": False,
        "stagev2_data_used_for_training": False,
        "stagev2_tail_data_used_for_training": False,
        "mu_window": [cfg.mu_min, cfg.mu_max],
        "excluded_nodes": ["gpuh01", "gpuh14"],
        "python": sys.executable,
        "confirm_stagev3_production": os.environ.get("CONFIRM_STAGEV3_PRODUCTION", "0"),
        "py_compile_returncode": pyc.returncode,
        "py_compile_stderr": pyc.stderr,
    }
    (args.output_dir / "stagev_v3_preflight.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "stagev_v3_preflight.md").write_text(
        "# Stage V-v3 Preflight\n\n"
        f"- status: `{status}`\n"
        f"- run_id: `{cfg.run_id}`\n"
        "- excluded_nodes: `gpuh01,gpuh14`\n"
        "- cold_start: `True`\n"
        "- stagev2_tail_data_used_for_training: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
