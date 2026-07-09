#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TARGET_REL = Path("ml_phase/stagev_acqv2.py")
SELECTOR_REL = Path("scripts/stagev_acqv2_select.py")

OLD_BLOCK = '''    cols = list(model["feature_columns"])
    x = features[cols].to_numpy(float)
'''

NEW_BLOCK = '''    cols = list(model["feature_columns"])
    frame = features.copy()
    for col in cols:
        if col not in frame:
            frame[col] = 0.0
    x = frame[cols].to_numpy(float)
'''


def find_package_root(explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if (root / TARGET_REL).exists():
            return root
        raise FileNotFoundError(f"cannot find {TARGET_REL} under explicit root: {root}")

    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / TARGET_REL).exists():
                return candidate
    raise FileNotFoundError(
        "cannot locate package root. Run from the Stage V package root or pass --root /path/to/package."
    )


def apply_patch(root: Path) -> str:
    target = root / TARGET_REL
    text = target.read_text(encoding="utf-8")

    if NEW_BLOCK in text:
        return "already_patched"
    if OLD_BLOCK not in text:
        context = ""
        marker = "def predict_linear_value_model"
        idx = text.find(marker)
        if idx >= 0:
            context = text[idx : idx + 600]
        raise RuntimeError(
            "patch target block not found and expected patched block is absent.\n"
            "Relevant local context:\n"
            f"{context}"
        )

    target.write_text(text.replace(OLD_BLOCK, NEW_BLOCK), encoding="utf-8", newline="\n")
    return "patched"


def verify_compile(root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "py_compile",
        str(root / TARGET_REL),
        str(root / SELECTOR_REL),
    ]
    subprocess.run(cmd, check=True)


def print_context(root: Path) -> None:
    target = root / TARGET_REL
    lines = target.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("def predict_linear_value_model"):
            start = max(i, 0)
            end = min(i + 12, len(lines))
            for lineno in range(start, end):
                print(f"{lineno + 1}: {lines[lineno]}")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Stage V iter002 reward-model prediction hotfix.")
    parser.add_argument("--root", default=None, help="Stage V package root. Defaults to auto-detect.")
    parser.add_argument("--no-compile", action="store_true", help="Skip py_compile verification.")
    args = parser.parse_args()

    root = find_package_root(args.root)
    status = apply_patch(root)
    if not args.no_compile:
        verify_compile(root)

    print(f"package_root={root}")
    print(f"hotfix_status={status}")
    print_context(root)
    print("")
    print("Resume command:")
    print("export CONFIRM_STAGEV_PRODUCTION=1")
    print(
        "START_ITER=2 nohup bash scripts/resume_stagev_acqv2_full_loop.sh "
        "> stagev_acqv2_boundary_support_learned_residual_3d_v1_resume_iter002.nohup.log 2>&1 &"
    )


if __name__ == "__main__":
    main()
