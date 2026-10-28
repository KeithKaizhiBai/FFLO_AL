#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${script_dir}/apply_stagev_acqv2_iter002_hotfix.py" "$@"
