from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "topo_trivial_full_loop_package"


def add(rows, check, status, detail):
    rows.append({"check": check, "status": status, "detail": detail})


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    add(rows, "topology_oracle_exists", "pass" if (ROOT / "ml_phase" / "topology_oracle.py").exists() else "fail", "ml_phase/topology_oracle.py")
    add(rows, "topology_pass_script_exists", "pass" if (ROOT / "scripts" / "run_topology_pass_dataset_iter035.py").exists() else "fail", "scripts/run_topology_pass_dataset_iter035.py")
    active_text = (ROOT / "ml_phase" / "active_refine.py").read_text(encoding="utf-8", errors="replace")
    add(rows, "sobol_initialization_supported", "pass" if "sobol_scrambled" in active_text else "fail", "active_refine initialization choices")
    has_topo_acq = ("A_spectral" in active_text) and ("topo_trivial" in active_text)
    add(rows, "topology_aware_acquisition_supported", "fail" if not has_topo_acq else "pass", "production active_refine must expose topology-aware acquisition before full-loop submission")
    submit_text = (ROOT / "scripts" / "submit_topo_trivial_full_loop.sh").read_text(encoding="utf-8", errors="replace")
    add(rows, "submit_script_excludes_gpuh01", "pass" if "gpuh01" in submit_text else "fail", "EXCLUDE_NODES=gpuh01")
    add(rows, "submit_script_uses_topo_trivial_profile", "pass" if 'ACQUISITION_PROFILE="topo_trivial"' in submit_text else "fail", "no silent phase-only fallback")
    add(rows, "submit_script_enables_topology_exact", "pass" if "--enable-topology-classification" in submit_text else "fail", "exact shards must write topology fields")

    summary_path = ROOT / "topology_reference" / "topology_pass_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        counts = {(str(r.get("thermo_phase")), str(r.get("topology_label"))): int(r.get("count", 0)) for r in summary.get("counts", [])}
        expected = {
            ("uniform_SC", "trivial"): 715,
            ("FFLO", "trivial"): 3127,
            ("FFLO", "topological"): 1515,
            ("FFLO", "gapless_SC"): 195,
            ("FFLO", "topology_unresolved"): 15,
        }
        for key, value in expected.items():
            got = counts.get(key, 0)
            add(rows, f"reference_count_{key[0]}_{key[1]}", "pass" if got == value else "fail", f"got={got}; expected={value}")
    else:
        add(rows, "reference_summary_exists", "fail", str(summary_path))

    shell_failures = []
    for path in sorted(ROOT.rglob("*.sh")):
        data = path.read_bytes()
        if data.startswith(b"\xef\xbb\xbf") or b"\r\n" in data or b"\r" in data:
            shell_failures.append(path.relative_to(ROOT).as_posix())
        try:
            data.decode("ascii")
        except UnicodeDecodeError:
            shell_failures.append(path.relative_to(ROOT).as_posix() + ":non_ascii")
    add(rows, "shell_scripts_ascii_lf_no_bom", "pass" if not shell_failures else "fail", ";".join(shell_failures[:10]))

    status = "pass" if all(r["status"] == "pass" for r in rows) else "blocked"
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "tables" / "package_preflight_checks.csv", rows)
    (OUT / "package_preflight.json").write_text(json.dumps({"status": status, "checks": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": rows}, indent=2))


if __name__ == "__main__":
    main()
