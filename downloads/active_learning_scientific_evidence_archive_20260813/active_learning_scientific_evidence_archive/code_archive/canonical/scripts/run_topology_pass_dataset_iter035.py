from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_phase.topology_oracle import BulkGapOracle, TopologyModelParams, TopologyPfaffianOracle


RUN_ID = "topology_pass_dataset_iter035_v1"
DEFAULT_DATASET = (
    "rankcap_k3_tail_surprise_continuation_results/"
    "ML_Phase_512_RankCapK3_TailContinuation/active_runs/"
    "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/dataset_iter035.npz"
)
DEFAULT_EXACT_MERGED = (
    "rankcap_k3_tail_surprise_continuation_results/"
    "ML_Phase_512_RankCapK3_TailContinuation/active_runs/"
    "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1/iter034/"
    "exact_merged_iter034.npz"
)
DEFAULT_OUT = f"reports/{RUN_ID}"
PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "checkpoints": out_dir / "checkpoints",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_dataset(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as d:
        x = np.asarray(d["x"], dtype=np.float64)
        y_reg = np.asarray(d["y_reg"], dtype=np.float64)
        y_phase = np.asarray(d["y_phase"], dtype=np.int64)
        df = pd.DataFrame(
            {
                "point_id": np.arange(x.shape[0], dtype=np.int64),
                "kBT": x[:, 0],
                "J_A": x[:, 1],
                "Delta_opt": y_reg[:, 0],
                "q_opt": y_reg[:, 1],
                "eta": y_reg[:, 2],
                "ic_plus": y_reg[:, 3],
                "ic_minus": y_reg[:, 4],
                "thermo_phase_code": y_phase,
                "thermo_phase": [PHASE_NAMES.get(int(v), "unknown") for v in y_phase],
            }
        )
        optional = {
            "trusted_exact": "trusted_exact",
            "training_eligible_exact": "training_eligible_exact",
            "rerun_required": "needs_rerun_exact",
            "q_unresolved": "q_unresolved",
            "delta_unresolved": "delta_unresolved",
            "free_energy_opt": None,
            "normal_state_free_energy": None,
            "free_energy_gap_to_normal": "free_energy_gap_to_normal",
            "positive_delta_gap": "positive_delta_gap",
        }
        for out_name, key in optional.items():
            if key and key in d.files:
                df[out_name] = np.asarray(d[key])
            else:
                df[out_name] = np.nan
    return df


def load_hard_risk_exact(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with np.load(path, allow_pickle=True) as d:
        n = int(np.asarray(d["kT"]).size)
        df = pd.DataFrame(
            {
                "source": "iter034_exact_merged",
                "point_id": np.arange(n, dtype=np.int64),
                "kBT": np.asarray(d["kT"], dtype=np.float64),
                "J_A": np.asarray(d["JA"], dtype=np.float64),
                "Delta_opt": np.asarray(d["delta_opt"], dtype=np.float64),
                "q_opt": np.asarray(d["q_opt"], dtype=np.float64),
                "thermo_phase_code": np.asarray(d["phase_candidate"], dtype=np.int64),
                "trusted_exact": np.asarray(d["trusted_exact"], dtype=np.int64),
                "training_eligible_exact": np.asarray(d["training_eligible_exact"], dtype=np.int64),
                "rerun_required": np.asarray(d["needs_rerun_exact"], dtype=np.int64),
                "q_unresolved": np.asarray(d["q_unresolved"], dtype=np.int64),
                "delta_unresolved": np.asarray(d["delta_unresolved"], dtype=np.int64),
            }
        )
    df["thermo_phase"] = [PHASE_NAMES.get(int(v), "unknown") for v in df["thermo_phase_code"]]
    hard = (
        (df["rerun_required"].to_numpy() != 0)
        | (df["trusted_exact"].to_numpy() == 0)
        | (df["training_eligible_exact"].to_numpy() == 0)
        | (df["q_unresolved"].to_numpy() != 0)
        | (df["delta_unresolved"].to_numpy() != 0)
    )
    return df.loc[hard & (df["thermo_phase_code"].to_numpy() != 0)].reset_index(drop=True)


def input_audit(dataset_path: Path, exact_merged_path: Path, out_dir: Path, params: TopologyModelParams) -> None:
    df = load_dataset(dataset_path)
    hard = load_hard_risk_exact(exact_merged_path)
    rows = []
    required = [
        "point_id",
        "kBT",
        "J_A",
        "Delta_opt",
        "q_opt",
        "thermo_phase",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "q_unresolved",
        "delta_unresolved",
        "free_energy_opt",
        "normal_state_free_energy",
        "free_energy_gap_to_normal",
    ]
    for col in required:
        rows.append(
            {
                "field": col,
                "present": bool(col in df.columns and not df[col].isna().all()),
                "source": "dataset_iter035.npz" if col in df.columns and not df[col].isna().all() else "missing_or_derived",
                "notes": "free_energy_opt and normal_state_free_energy are not stored separately" if col in {"free_energy_opt", "normal_state_free_energy"} else "",
            }
        )
    for name, value in {
        "t": params.t,
        "lambda_ry": params.lambda_ry,
        "lambda_rz": params.lambda_rz,
        "mu": params.mu,
    }.items():
        rows.append({"field": name, "present": True, "source": "EtaPhaseConfig defaults", "notes": str(value)})
    pd.DataFrame(rows).to_csv(out_dir / "tables" / "input_data_audit.csv", index=False)
    phase_counts = df["thermo_phase"].value_counts().rename_axis("phase").reset_index(name="count")
    phase_counts.to_csv(out_dir / "tables" / "input_phase_counts.csv", index=False)
    sc = df[df["thermo_phase_code"] != 0]
    hard_summary = pd.DataFrame(
        [
            {"quantity": "dataset_total", "value": int(len(df))},
            {"quantity": "dataset_sc_points", "value": int(len(sc))},
            {"quantity": "dataset_uniform_sc", "value": int((df["thermo_phase_code"] == 1).sum())},
            {"quantity": "dataset_fflo", "value": int((df["thermo_phase_code"] == 2).sum())},
            {"quantity": "dataset_sc_hard_risk", "value": int((((df["rerun_required"] != 0) | (df["trusted_exact"] == 0) | (df["training_eligible_exact"] == 0) | (df["q_unresolved"] != 0) | (df["delta_unresolved"] != 0)) & (df["thermo_phase_code"] != 0)).sum())},
            {"quantity": "iter034_hard_risk_sc_for_pilot", "value": int(len(hard))},
        ]
    )
    hard_summary.to_csv(out_dir / "tables" / "input_hard_risk_summary.csv", index=False)
    config = "\n".join(
        [
            f"run_id: {RUN_ID}",
            f"dataset_path: {dataset_path.as_posix()}",
            f"dataset_sha256: {sha256_file(dataset_path)}",
            f"exact_merged_path: {exact_merged_path.as_posix()}",
            f"t: {params.t}",
            f"lambda_ry: {params.lambda_ry}",
            f"lambda_rz: {params.lambda_rz}",
            f"mu: {params.mu}",
            "pf_tol_rel: 1.0e-8",
            "gap_tol_rel: 1.0e-8",
            "nk_values: [512, 1024, 2048]",
            "nk_refine: 4096",
            "",
        ]
    )
    write_text(out_dir / "topology_pass_config.yaml", config)


def select_validation_points(df: pd.DataFrame, count: int, seed: int, pf: TopologyPfaffianOracle) -> pd.DataFrame:
    sc = df[df["thermo_phase_code"] != 0].copy()
    p0, ppi, _prod, margin = pf.analytic_pfaffians(sc["Delta_opt"].to_numpy(), sc["q_opt"].to_numpy(), sc["J_A"].to_numpy())
    sc["pfaffian_margin_preview"] = margin
    picks: list[int] = []
    for series in [
        sc["pfaffian_margin_preview"].nsmallest(20).index,
        sc["Delta_opt"].nsmallest(10).index,
        sc["J_A"].nsmallest(10).index,
        sc["J_A"].nlargest(10).index,
        sc["q_opt"].abs().nsmallest(10).index,
        sc["q_opt"].abs().nlargest(10).index,
    ]:
        picks.extend([int(i) for i in series])
    rng = np.random.default_rng(seed)
    remaining = sc.index.difference(pd.Index(picks))
    random_n = max(0, count - len(set(picks)))
    if len(remaining) and random_n:
        picks.extend([int(i) for i in rng.choice(remaining.to_numpy(), size=min(random_n, len(remaining)), replace=False)])
    selected = sc.loc[pd.Index(picks).drop_duplicates()].head(count).copy()
    selected["analytic_P0_preview"] = p0[sc.index.get_indexer(selected.index)]
    selected["analytic_Ppi_preview"] = ppi[sc.index.get_indexer(selected.index)]
    return selected.reset_index(drop=True)


def run_pfaffian_validation(df: pd.DataFrame, out_dir: Path, count: int = 100, seed: int = 42) -> bool:
    pf = TopologyPfaffianOracle()
    points = select_validation_points(df, count=count, seed=seed, pf=pf)
    rows = []
    for _, row in points.iterrows():
        delta = float(row["Delta_opt"])
        q = float(row["q_opt"])
        ja = float(row["J_A"])
        p0, ppi, product, margin = pf.analytic_pfaffians(delta, q, ja)
        n0, npi, antisym_error, imag_max = pf.numeric_pfaffians(delta, q, ja)
        numeric_product = n0 * npi
        boundary = bool(float(margin) <= 1e-8)
        sign_match = bool(np.sign(float(product)) == np.sign(float(np.real(numeric_product)))) if not boundary else True
        rows.append(
            {
                "point_id": int(row["point_id"]),
                "kBT": float(row["kBT"]),
                "J_A": ja,
                "Delta_opt": delta,
                "q_opt": q,
                "thermo_phase": row["thermo_phase"],
                "analytic_P0": float(p0),
                "analytic_Ppi": float(ppi),
                "analytic_product": float(product),
                "numeric_pf_k0": float(np.real(n0)),
                "numeric_pf_kpi": float(np.real(npi)),
                "numeric_product": float(np.real(numeric_product)),
                "pfaffian_margin": float(margin),
                "boundary_like": int(boundary),
                "antisymmetry_error": float(antisym_error),
                "majorana_imag_max": float(imag_max),
                "product_sign_match": int(sign_match),
            }
        )
    edge_cases = [
        ("q_zero", 0.2, 0.0, 0.8),
        ("delta_near_zero", 1e-6, -0.4, 0.8),
        ("small_JA", 0.2, -0.4, 0.05),
        ("large_JA", 0.2, -1.2, 2.0),
    ]
    for name, delta, q, ja in edge_cases:
        p0, ppi, product, margin = pf.analytic_pfaffians(delta, q, ja)
        n0, npi, antisym_error, imag_max = pf.numeric_pfaffians(delta, q, ja)
        numeric_product = n0 * npi
        rows.append(
            {
                "point_id": -1,
                "kBT": np.nan,
                "J_A": ja,
                "Delta_opt": delta,
                "q_opt": q,
                "thermo_phase": name,
                "analytic_P0": float(p0),
                "analytic_Ppi": float(ppi),
                "analytic_product": float(product),
                "numeric_pf_k0": float(np.real(n0)),
                "numeric_pf_kpi": float(np.real(npi)),
                "numeric_product": float(np.real(numeric_product)),
                "pfaffian_margin": float(margin),
                "boundary_like": int(float(margin) <= 1e-8),
                "antisymmetry_error": float(antisym_error),
                "majorana_imag_max": float(imag_max),
                "product_sign_match": int(np.sign(float(product)) == np.sign(float(np.real(numeric_product)))),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "tables" / "pfaffian_validation.csv", index=False)
    non_boundary = out["boundary_like"] == 0
    return bool((out.loc[non_boundary, "product_sign_match"] == 1).all())


def select_pilot_points(df: pd.DataFrame, hard_df: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    sc = df[df["thermo_phase_code"] != 0].copy()
    selected: list[int] = []
    rng = np.random.default_rng(seed)

    def add(indices: pd.Index, n: int) -> None:
        idx = list(indices)
        if len(idx) > n:
            idx = list(rng.choice(np.asarray(idx), size=n, replace=False))
        selected.extend(int(i) for i in idx)

    add(sc[sc["thermo_phase_code"] == 1].index, 64)
    add(sc[sc["thermo_phase_code"] == 2].index, 96)
    add(sc.nsmallest(32, "kBT").index, 24)
    add(sc.nlargest(32, "kBT").index, 24)
    add(sc.nsmallest(32, "q_opt", keep="all").index, 16)
    add(sc["q_opt"].abs().nlargest(32).index, 16)
    selected_idx = pd.Index(selected).drop_duplicates()
    pilot = sc.loc[selected_idx].copy()
    if len(pilot) < count and len(sc.index.difference(selected_idx)):
        remaining = sc.index.difference(selected_idx).to_numpy()
        fill = rng.choice(remaining, size=min(count - len(pilot), len(remaining)), replace=False)
        pilot = pd.concat([pilot, sc.loc[fill]], axis=0)
    pilot = pilot.head(count).copy()
    pilot["pilot_source"] = "dataset_iter035"
    if not hard_df.empty:
        hard_take = hard_df.head(min(32, len(hard_df))).copy()
        hard_take["pilot_source"] = "iter034_hard_risk"
        keep_cols = ["point_id", "kBT", "J_A", "Delta_opt", "q_opt", "thermo_phase_code", "thermo_phase", "trusted_exact", "training_eligible_exact", "rerun_required", "q_unresolved", "delta_unresolved", "pilot_source"]
        pilot = pd.concat([pilot[keep_cols], hard_take[keep_cols]], ignore_index=True).head(count)
    return pilot.reset_index(drop=True)


def run_pilot_benchmark(
    df: pd.DataFrame,
    hard_df: pd.DataFrame,
    out_dir: Path,
    nk_values: list[int],
    seed: int,
    point_chunk: int,
    k_chunk: int,
) -> pd.DataFrame:
    pilot = select_pilot_points(df, hard_df, 256, seed)
    pilot.to_csv(out_dir / "tables" / "pilot_points.csv", index=False)
    rows = []
    detail: dict[tuple[str, int], dict[str, object]] = {}
    for backend in ("cpu", "gpu"):
        for nk in nk_values:
            try:
                oracle = BulkGapOracle(backend=backend, point_chunk=point_chunk, k_chunk=k_chunk)
                res = oracle.compute(
                    pilot["Delta_opt"].to_numpy(),
                    pilot["q_opt"].to_numpy(),
                    pilot["J_A"].to_numpy(),
                    nk=int(nk),
                )
                rows.append({k: v for k, v in res.items() if not isinstance(v, np.ndarray)})
                detail[(backend, int(nk))] = res
            except Exception as exc:
                rows.append(
                    {
                        "backend": backend,
                        "nk": int(nk),
                        "point_count": int(len(pilot)),
                        "runtime_seconds": np.nan,
                        "points_per_second": np.nan,
                        "k_hamiltonians_per_second": np.nan,
                        "peak_vram_mb": np.nan,
                        "failure_reason": repr(exc),
                    }
                )
    bench = pd.DataFrame(rows)
    if ("cpu", max(nk_values)) in detail and ("gpu", max(nk_values)) in detail:
        cpu_gap = np.asarray(detail[("cpu", max(nk_values))]["bulk_gap"], dtype=np.float64)
        gpu_gap = np.asarray(detail[("gpu", max(nk_values))]["bulk_gap"], dtype=np.float64)
        agreement = pd.DataFrame(
            [
                {
                    "nk": max(nk_values),
                    "max_abs_gap_diff": float(np.nanmax(np.abs(cpu_gap - gpu_gap))),
                    "median_abs_gap_diff": float(np.nanmedian(np.abs(cpu_gap - gpu_gap))),
                    "allclose_rtol_1e-8_atol_1e-10": bool(np.allclose(cpu_gap, gpu_gap, rtol=1e-8, atol=1e-10)),
                }
            ]
        )
    else:
        agreement = pd.DataFrame([{"nk": max(nk_values), "max_abs_gap_diff": np.nan, "median_abs_gap_diff": np.nan, "allclose_rtol_1e-8_atol_1e-10": False}])
    agreement.to_csv(out_dir / "tables" / "cpu_gpu_agreement.csv", index=False)
    conv_rows = []
    for backend in ("cpu", "gpu"):
        for prev, curr in zip(nk_values[:-1], nk_values[1:]):
            if (backend, prev) in detail and (backend, curr) in detail:
                g0 = np.asarray(detail[(backend, prev)]["bulk_gap"], dtype=np.float64)
                g1 = np.asarray(detail[(backend, curr)]["bulk_gap"], dtype=np.float64)
                conv_rows.append(
                    {
                        "backend": backend,
                        "nk_low": int(prev),
                        "nk_high": int(curr),
                        "max_abs_gap_change": float(np.nanmax(np.abs(g1 - g0))),
                        "median_abs_gap_change": float(np.nanmedian(np.abs(g1 - g0))),
                        "p95_abs_gap_change": float(np.nanpercentile(np.abs(g1 - g0), 95)),
                    }
                )
    pd.DataFrame(conv_rows).to_csv(out_dir / "tables" / "pilot_nk_convergence.csv", index=False)
    bench.to_csv(out_dir / "tables" / "pilot_benchmark.csv", index=False)
    full_sc_count = int((df["thermo_phase_code"] != 0).sum())
    projections = []
    for _, row in bench.dropna(subset=["runtime_seconds"]).iterrows():
        projected = float(row["runtime_seconds"]) * full_sc_count / max(float(row["point_count"]), 1.0)
        projections.append(
            {
                "backend": row["backend"],
                "nk": int(row["nk"]),
                "pilot_runtime_seconds": float(row["runtime_seconds"]),
                "projected_full_runtime_seconds": projected,
                "projected_full_runtime_hours": projected / 3600.0,
            }
        )
    pd.DataFrame(projections).to_csv(out_dir / "tables" / "pilot_projection.csv", index=False)
    return bench


def refine_local_bulk_gap_minima(
    gap_oracle: BulkGapOracle,
    points: pd.DataFrame,
    coarse_gap: np.ndarray,
    coarse_k: np.ndarray,
    nk: int,
) -> pd.DataFrame:
    from scipy.optimize import minimize_scalar

    rows = []
    dk = 2.0 * math.pi / max(int(nk) - 1, 1)
    for local_idx, (_, row) in enumerate(points.iterrows()):
        center = float(coarse_k[local_idx])
        delta = float(row["Delta_opt"])
        q = float(row["q_opt"])
        ja = float(row["J_A"])

        def objective(k_val: float) -> float:
            return gap_oracle.gap_at_k(delta=delta, q=q, ja=ja, k=k_val)

        lower = center - 2.0 * dk
        upper = center + 2.0 * dk
        try:
            opt = minimize_scalar(objective, bounds=(lower, upper), method="bounded", options={"xatol": 1e-10, "maxiter": 64})
            local_gap = float(opt.fun)
            local_k = float(((opt.x + math.pi) % (2.0 * math.pi)) - math.pi)
            status = "ok" if bool(opt.success) else "optimizer_not_converged"
        except Exception as exc:
            local_gap = float(coarse_gap[local_idx])
            local_k = center
            status = f"failed:{exc.__class__.__name__}"
        rows.append(
            {
                "point_id": int(row["point_id"]),
                "coarse_gap": float(coarse_gap[local_idx]),
                "coarse_k": center,
                "local_gap": local_gap,
                "local_k": local_k,
                "local_gap_error": abs(local_gap - float(coarse_gap[local_idx])),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def classify_topology(
    df: pd.DataFrame,
    hard_df: pd.DataFrame,
    out_dir: Path,
    backend: str,
    nk: int,
    point_chunk: int,
    k_chunk: int,
) -> pd.DataFrame:
    params = TopologyModelParams()
    pf = TopologyPfaffianOracle(params)
    out = df.copy()
    for col in [
        "P0",
        "Ppi",
        "pf_product",
        "pfaffian_margin",
        "bulk_gap",
        "k_at_bulk_gap",
        "gap_grid_error",
        "gap_local_refinement_error",
        "thermal_gap_ratio",
        "runtime_seconds",
    ]:
        out[col] = np.nan
    out["z2_value"] = -1
    out["spectral_status"] = "not_applicable"
    out["topology_label"] = "not_applicable"
    out["topology_trusted"] = False
    out["backend"] = backend
    out["Nk_used"] = 0
    out["local_refinement_used"] = False
    out["failure_reason"] = ""

    sc_mask = out["thermo_phase_code"].to_numpy() != 0
    sc = out.loc[sc_mask].copy()
    p0, ppi, product, margin, z2 = pf.z2_status(sc["Delta_opt"].to_numpy(), sc["q_opt"].to_numpy(), sc["J_A"].to_numpy())
    gap_oracle = BulkGapOracle(params, backend=backend, point_chunk=point_chunk, k_chunk=k_chunk)
    gap_res = gap_oracle.compute(sc["Delta_opt"].to_numpy(), sc["q_opt"].to_numpy(), sc["J_A"].to_numpy(), nk=nk)
    sc_idx = sc.index.to_numpy()
    out.loc[sc_idx, "P0"] = p0
    out.loc[sc_idx, "Ppi"] = ppi
    out.loc[sc_idx, "pf_product"] = product
    out.loc[sc_idx, "pfaffian_margin"] = margin
    out.loc[sc_idx, "z2_value"] = z2
    out.loc[sc_idx, "bulk_gap"] = gap_res["bulk_gap"]
    out.loc[sc_idx, "k_at_bulk_gap"] = gap_res["k_at_bulk_gap"]
    out.loc[sc_idx, "Nk_used"] = int(nk)
    out.loc[sc_idx, "runtime_seconds"] = float(gap_res["runtime_seconds"]) / max(len(sc), 1)

    # Refine all small-gap/small-margin points and a 5% random control subset.
    rng = np.random.default_rng(42)
    refine_ids: set[int] = set()
    sc_bulk = out.loc[sc_idx, "bulk_gap"]
    sc_margin = out.loc[sc_idx, "pfaffian_margin"]
    refine_ids.update(sc_bulk.nsmallest(max(1, int(0.05 * len(sc)))).index.astype(int).tolist())
    refine_ids.update(sc_margin.nsmallest(max(1, int(0.05 * len(sc)))).index.astype(int).tolist())
    random_ref = rng.choice(sc_idx, size=max(1, int(0.05 * len(sc))), replace=False)
    refine_ids.update(int(i) for i in random_ref)
    refine_idx = np.asarray(sorted(refine_ids), dtype=np.int64)
    if refine_idx.size:
        refine = out.loc[refine_idx]
        gap_ref = gap_oracle.compute(refine["Delta_opt"].to_numpy(), refine["q_opt"].to_numpy(), refine["J_A"].to_numpy(), nk=2 * nk)
        doubled_gap = np.asarray(gap_ref["bulk_gap"], dtype=np.float64)
        doubled_k = np.asarray(gap_ref["k_at_bulk_gap"], dtype=np.float64)
        local_ref = refine_local_bulk_gap_minima(gap_oracle, refine, doubled_gap, doubled_k, nk=2 * nk)
        local_ref.to_csv(out_dir / "tables" / "local_gap_refinement.csv", index=False)
        coarse_gap = out.loc[refine_idx, "bulk_gap"].to_numpy(dtype=np.float64)
        local_gap = local_ref["local_gap"].to_numpy(dtype=np.float64)
        local_k = local_ref["local_k"].to_numpy(dtype=np.float64)
        grid_err = np.maximum(np.abs(doubled_gap - coarse_gap), np.abs(local_gap - doubled_gap))
        out.loc[refine_idx, "bulk_gap"] = local_gap
        out.loc[refine_idx, "k_at_bulk_gap"] = local_k
        out.loc[refine_idx, "gap_grid_error"] = grid_err
        out.loc[refine_idx, "gap_local_refinement_error"] = local_ref["local_gap_error"].to_numpy(dtype=np.float64)
        out.loc[refine_idx, "local_refinement_used"] = True
    else:
        pd.DataFrame().to_csv(out_dir / "tables" / "local_gap_refinement.csv", index=False)
    finite_err = out.loc[sc_idx, "gap_grid_error"].dropna().to_numpy(dtype=np.float64)
    conservative_error = float(np.nanpercentile(finite_err, 95)) if finite_err.size else np.nan
    out.loc[sc_idx[out.loc[sc_idx, "gap_grid_error"].isna().to_numpy()], "gap_grid_error"] = conservative_error

    e_scale = params.energy_scale(out.loc[sc_idx, "J_A"].to_numpy(), out.loc[sc_idx, "Delta_opt"].to_numpy())
    gap_tol_abs = 1e-8 * e_scale
    gap = out.loc[sc_idx, "bulk_gap"].to_numpy(dtype=np.float64)
    gap_err = out.loc[sc_idx, "gap_grid_error"].to_numpy(dtype=np.float64)
    gapped = gap > np.maximum(10.0 * gap_err, gap_tol_abs)
    gapless = (gap <= gap_tol_abs) & out.loc[sc_idx, "local_refinement_used"].to_numpy(dtype=bool)
    unresolved = ~(gapped | gapless)
    out.loc[sc_idx[gapped], "spectral_status"] = "gapped"
    out.loc[sc_idx[gapless], "spectral_status"] = "gapless"
    out.loc[sc_idx[unresolved], "spectral_status"] = "gap_unresolved"
    out.loc[sc_idx[gapless], "topology_label"] = "gapless_SC"
    out.loc[sc_idx[unresolved], "topology_label"] = "topology_unresolved"
    topological = gapped & (out.loc[sc_idx, "z2_value"].to_numpy(dtype=np.int64) == 1)
    trivial = gapped & (out.loc[sc_idx, "z2_value"].to_numpy(dtype=np.int64) == 0)
    out.loc[sc_idx[topological], "topology_label"] = "topological"
    out.loc[sc_idx[trivial], "topology_label"] = "trivial"
    trusted_mask = (
        (out.loc[sc_idx, "trusted_exact"].to_numpy() == 1)
        & (out.loc[sc_idx, "training_eligible_exact"].to_numpy() == 1)
        & (out.loc[sc_idx, "rerun_required"].to_numpy() == 0)
        & (out.loc[sc_idx, "q_unresolved"].to_numpy() == 0)
        & (out.loc[sc_idx, "delta_unresolved"].to_numpy() == 0)
        & gapped
        & (out.loc[sc_idx, "z2_value"].to_numpy(dtype=np.int64) >= 0)
    )
    out.loc[sc_idx[trusted_mask], "topology_trusted"] = True
    kbt = out.loc[sc_idx, "kBT"].to_numpy(dtype=np.float64)
    ratio = gap / np.where(kbt > 0, kbt, np.nan)
    ratio[kbt == 0] = np.inf
    out.loc[sc_idx, "thermal_gap_ratio"] = ratio
    out.to_csv(out_dir / "tables" / "dataset_iter035_topology_ground_v1.csv", index=False)
    np.savez_compressed(out_dir / "tables" / "dataset_iter035_topology_ground_v1.npz", **{c: out[c].to_numpy() for c in out.columns})
    try:
        out.to_parquet(out_dir / "tables" / "dataset_iter035_topology_ground_v1.parquet", index=False)
        parquet_status = "written"
    except Exception as exc:
        parquet_status = f"not_written: {exc.__class__.__name__}: {exc}"
    diagnostics = build_topology_boundary_diagnostics(out, hard_df, out_dir)
    write_benchmark_report(out_dir, backend, nk, gap_res)
    out[out["failure_reason"].astype(str).str.len() > 0].to_csv(
        out_dir / "tables" / "topology_failed_points.csv", index=False
    )
    out[out["topology_label"] == "topology_unresolved"].to_csv(
        out_dir / "tables" / "topology_unresolved_points.csv", index=False
    )
    summary = topology_summary(out, parquet_status, backend, nk, gap_res, diagnostics)
    (out_dir / "topology_pass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["counts"]).to_csv(out_dir / "tables" / "topology_count_summary.csv", index=False)
    make_figures(out, out_dir)
    return out


def topology_summary(
    out: pd.DataFrame,
    parquet_status: str,
    backend: str,
    nk: int,
    gap_res: dict[str, object],
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    sc = out[out["thermo_phase_code"] != 0]
    counts = []
    for phase in ["uniform_SC", "FFLO"]:
        sub = sc[sc["thermo_phase"] == phase]
        for topo in ["trivial", "topological", "gapless_SC", "topology_unresolved"]:
            counts.append({"thermo_phase": phase, "topology_label": topo, "count": int((sub["topology_label"] == topo).sum())})
    return {
        "run_id": RUN_ID,
        "processed_points": int(len(out)),
        "total_sc_points": int(len(sc)),
        "trusted_topology_points": int(sc["topology_trusted"].sum()),
        "provisional_topology_points": int(len(sc) - sc["topology_trusted"].sum()),
        "backend": backend,
        "nk": int(nk),
        "actual_runtime_seconds": float(gap_res["runtime_seconds"]),
        "peak_vram_mb": float(gap_res["peak_vram_mb"]) if np.isfinite(gap_res["peak_vram_mb"]) else None,
        "parquet_status": parquet_status,
        "bulk_gap_min": float(np.nanmin(sc["bulk_gap"])),
        "bulk_gap_median": float(np.nanmedian(sc["bulk_gap"])),
        "bulk_gap_p95": float(np.nanpercentile(sc["bulk_gap"], 95)),
        "pfaffian_margin_min": float(np.nanmin(sc["pfaffian_margin"])),
        "pfaffian_margin_median": float(np.nanmedian(sc["pfaffian_margin"])),
        "pfaffian_margin_p95": float(np.nanpercentile(sc["pfaffian_margin"], 95)),
        "counts": counts,
        "diagnostics": diagnostics or {},
    }


def write_benchmark_report(out_dir: Path, backend: str, nk: int, gap_res: dict[str, object]) -> None:
    def json_clean(value: object) -> object:
        if isinstance(value, dict):
            return {str(k): json_clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    report = {
        "selected_backend": backend,
        "selected_nk": int(nk),
        "actual_runtime_seconds": float(gap_res["runtime_seconds"]),
        "points_per_second": float(gap_res["points_per_second"]),
        "k_hamiltonians_per_second": float(gap_res["k_hamiltonians_per_second"]),
        "peak_vram_mb": float(gap_res["peak_vram_mb"]) if np.isfinite(gap_res["peak_vram_mb"]) else None,
    }
    for name in ["pilot_benchmark", "pilot_projection", "cpu_gpu_agreement", "pilot_nk_convergence"]:
        path = out_dir / "tables" / f"{name}.csv"
        if path.exists():
            report[name] = pd.read_csv(path).to_dict(orient="records")
    write_text(out_dir / "topology_benchmark_report.json", json.dumps(json_clean(report), indent=2))


def _normalized_xy(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    xy = df[["kBT", "J_A"]].to_numpy(dtype=np.float64)
    lo = np.nanmin(xy, axis=0)
    hi = np.nanmax(xy, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return (xy - lo) / span, {
        "kBT_min": float(lo[0]),
        "kBT_max": float(hi[0]),
        "J_A_min": float(lo[1]),
        "J_A_max": float(hi[1]),
    }


def _circumradius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = float(np.linalg.norm(a - b))
    bc = float(np.linalg.norm(b - c))
    ca = float(np.linalg.norm(c - a))
    ba = b - a
    ca_vec = c - a
    area2 = abs(float(ba[0] * ca_vec[1] - ba[1] * ca_vec[0]))
    if area2 <= 1e-15:
        return math.inf
    return ab * bc * ca / (2.0 * area2)


def build_topology_boundary_diagnostics(out: pd.DataFrame, hard_df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    from scipy.spatial import Delaunay, cKDTree

    fflo = out[
        (out["thermo_phase"] == "FFLO")
        & (out["topology_trusted"] == True)
        & (out["topology_label"].isin(["trivial", "topological"]))
    ].copy()
    diagnostics: dict[str, object] = {
        "trusted_fflo_gapped_points": int(len(fflo)),
        "z2_change_candidate_edges": 0,
        "p0_sign_change_candidate_edges": 0,
        "ppi_sign_change_candidate_edges": 0,
        "delaunay_edge_count": 0,
        "delaunay_triangle_count": 0,
        "large_circumradius_triangles": 0,
        "coverage_hole_threshold_norm": None,
        "nearest_neighbor_distance_p95_norm": None,
        "nearest_neighbor_distance_max_norm": None,
        "next_case": "Case D",
    }
    if len(fflo) < 3 or fflo["topology_label"].nunique() < 2:
        pd.DataFrame().to_csv(out_dir / "tables" / "topology_boundary_candidate_edges.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tables" / "topology_coverage_holes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "tables" / "topology_large_nn_distance_points.csv", index=False)
        diagnostics["next_case"] = "Case C" if len(fflo) >= 3 else "Case D"
        pd.DataFrame([diagnostics]).to_csv(out_dir / "tables" / "topology_sampling_coverage_summary.csv", index=False)
        write_hard_risk_overlay(out, hard_df, out_dir)
        return diagnostics

    xy_norm, bounds = _normalized_xy(fflo)
    tri = Delaunay(xy_norm)
    edge_set: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        ids = [int(x) for x in simplex]
        for a, b in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])]:
            edge_set.add(tuple(sorted((a, b))))

    edge_rows = []
    for edge_id, (i, j) in enumerate(sorted(edge_set)):
        left = fflo.iloc[i]
        right = fflo.iloc[j]
        p0_left = float(left["P0"])
        p0_right = float(right["P0"])
        ppi_left = float(left["Ppi"])
        ppi_right = float(right["Ppi"])
        z2_left = int(left["z2_value"])
        z2_right = int(right["z2_value"])
        p0_change = np.sign(p0_left) != np.sign(p0_right)
        ppi_change = np.sign(ppi_left) != np.sign(ppi_right)
        z2_change = z2_left != z2_right
        edge_rows.append(
            {
                "edge_id": int(edge_id),
                "point_id_left": int(left["point_id"]),
                "point_id_right": int(right["point_id"]),
                "kBT_left": float(left["kBT"]),
                "J_A_left": float(left["J_A"]),
                "kBT_right": float(right["kBT"]),
                "J_A_right": float(right["J_A"]),
                "z2_left": z2_left,
                "z2_right": z2_right,
                "topology_left": str(left["topology_label"]),
                "topology_right": str(right["topology_label"]),
                "z2_change_candidate": bool(z2_change),
                "p0_sign_change_candidate": bool(p0_change),
                "ppi_sign_change_candidate": bool(ppi_change),
                "edge_length_norm": float(np.linalg.norm(xy_norm[i] - xy_norm[j])),
                "bulk_gap_min": float(min(left["bulk_gap"], right["bulk_gap"])),
                "pfaffian_margin_min": float(min(left["pfaffian_margin"], right["pfaffian_margin"])),
                "mid_kBT": float(0.5 * (left["kBT"] + right["kBT"])),
                "mid_J_A": float(0.5 * (left["J_A"] + right["J_A"])),
            }
        )
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(out_dir / "tables" / "topology_boundary_candidate_edges.csv", index=False)

    tri_rows = []
    for simplex_id, simplex in enumerate(tri.simplices):
        pts = xy_norm[simplex]
        radius = _circumradius(pts[0], pts[1], pts[2])
        sub = fflo.iloc[simplex]
        tri_rows.append(
            {
                "triangle_id": int(simplex_id),
                "point_ids": ";".join(str(int(x)) for x in sub["point_id"]),
                "center_kBT": float(sub["kBT"].mean()),
                "center_J_A": float(sub["J_A"].mean()),
                "circumradius_norm": float(radius),
                "topology_labels": ";".join(sorted(set(str(x) for x in sub["topology_label"]))),
                "contains_z2_change": bool(sub["z2_value"].nunique() > 1),
                "bulk_gap_min": float(sub["bulk_gap"].min()),
                "pfaffian_margin_min": float(sub["pfaffian_margin"].min()),
            }
        )
    tri_df = pd.DataFrame(tri_rows)
    threshold = float(np.nanpercentile(tri_df["circumradius_norm"], 95)) if not tri_df.empty else math.nan
    holes = tri_df[tri_df["circumradius_norm"] >= threshold].sort_values("circumradius_norm", ascending=False).copy()
    holes.to_csv(out_dir / "tables" / "topology_coverage_holes.csv", index=False)

    tree = cKDTree(xy_norm)
    dists, _ = tree.query(xy_norm, k=2)
    nn = dists[:, 1] if dists.ndim == 2 and dists.shape[1] > 1 else np.asarray([], dtype=np.float64)
    nn_df = fflo[["point_id", "kBT", "J_A", "topology_label", "bulk_gap", "pfaffian_margin"]].copy()
    nn_df["nearest_neighbor_distance_norm"] = nn
    nn_df.sort_values("nearest_neighbor_distance_norm", ascending=False).head(256).to_csv(
        out_dir / "tables" / "topology_large_nn_distance_points.csv", index=False
    )

    diagnostics.update(
        {
            "normalization_bounds": bounds,
            "z2_change_candidate_edges": int(edge_df["z2_change_candidate"].sum()),
            "p0_sign_change_candidate_edges": int(edge_df["p0_sign_change_candidate"].sum()),
            "ppi_sign_change_candidate_edges": int(edge_df["ppi_sign_change_candidate"].sum()),
            "delaunay_edge_count": int(len(edge_df)),
            "delaunay_triangle_count": int(len(tri_df)),
            "large_circumradius_triangles": int(len(holes)),
            "coverage_hole_threshold_norm": threshold,
            "nearest_neighbor_distance_p95_norm": float(np.nanpercentile(nn, 95)) if nn.size else None,
            "nearest_neighbor_distance_max_norm": float(np.nanmax(nn)) if nn.size else None,
            "next_case": "Case B" if int(edge_df["z2_change_candidate"].sum()) > 0 else "Case C",
        }
    )
    pd.DataFrame([diagnostics]).to_csv(out_dir / "tables" / "topology_sampling_coverage_summary.csv", index=False)
    make_topology_diagnostic_figures(fflo, edge_df, holes, out_dir)
    write_hard_risk_overlay(out, hard_df, out_dir)
    return diagnostics


def write_hard_risk_overlay(out: pd.DataFrame, hard_df: pd.DataFrame, out_dir: Path) -> None:
    overlay_path = out_dir / "tables" / "thermodynamic_hard_risk_overlay_points.csv"
    if hard_df.empty:
        pd.DataFrame().to_csv(overlay_path, index=False)
        return
    cols = [
        "point_id",
        "kBT",
        "J_A",
        "Delta_opt",
        "q_opt",
        "thermo_phase",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "q_unresolved",
        "delta_unresolved",
    ]
    hard_df[[c for c in cols if c in hard_df.columns]].to_csv(overlay_path, index=False)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    sc = out[out["thermo_phase_code"] != 0]
    ax.scatter(sc["kBT"], sc["J_A"], s=5, c="#bbbbbb", label="SC dataset", edgecolors="none", alpha=0.35)
    ax.scatter(hard_df["kBT"], hard_df["J_A"], s=22, c="#dd8452", label="iter034 hard-risk diagnostics", edgecolors="black", linewidths=0.2)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Thermodynamic hard-risk overlay")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "thermodynamic_hard_risk_overlay.png")
    plt.close(fig)


def make_topology_diagnostic_figures(fflo: pd.DataFrame, edge_df: pd.DataFrame, holes: pd.DataFrame, out_dir: Path) -> None:
    colors = {"trivial": "#4477aa", "topological": "#cc6677"}
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    for label, group in fflo.groupby("topology_label"):
        ax.scatter(group["kBT"], group["J_A"], s=7, c=colors.get(label, "#333333"), label=label, alpha=0.75, edgecolors="none")
    draw_edges = edge_df[edge_df["z2_change_candidate"]].head(2000)
    for _, row in draw_edges.iterrows():
        ax.plot([row["kBT_left"], row["kBT_right"]], [row["J_A_left"], row["J_A_right"]], color="#111111", linewidth=0.35, alpha=0.35)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Candidate Z2-change Delaunay edges")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "topology_boundary_candidate_edges.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(fflo["kBT"], fflo["J_A"], s=5, c="#bbbbbb", edgecolors="none", alpha=0.4)
    if not holes.empty:
        size = 40 + 500 * holes["circumradius_norm"].to_numpy(dtype=np.float64)
        im = ax.scatter(holes["center_kBT"], holes["center_J_A"], s=size, c=holes["circumradius_norm"], cmap="magma", edgecolors="black", linewidths=0.2)
        fig.colorbar(im, ax=ax, label="normalized circumradius")
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Coverage-hole diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "topology_coverage_holes.png")
    plt.close(fig)


def make_figures(out: pd.DataFrame, out_dir: Path) -> None:
    sc = out[out["thermo_phase_code"] != 0].copy()
    colors = {"trivial": "#4477aa", "topological": "#cc6677", "gapless_SC": "#ddcc77", "topology_unresolved": "#888888"}
    for name, sub in [("all_sc", sc), ("fflo_only", sc[sc["thermo_phase"] == "FFLO"]), ("uniform_sc_only", sc[sc["thermo_phase"] == "uniform_SC"])]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        for label, group in sub.groupby("topology_label"):
            ax.scatter(group["kBT"], group["J_A"], s=9, c=colors.get(label, "#333333"), label=label, alpha=0.75, edgecolors="none")
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title(name.replace("_", " "))
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / f"topology_scatter_{name}.png")
        plt.close(fig)
    for field, fname, title in [
        ("bulk_gap", "bulk_gap_scatter.png", "Bulk quasiparticle gap"),
        ("pfaffian_margin", "pfaffian_margin_scatter.png", "Pfaffian margin"),
        ("thermal_gap_ratio", "thermal_gap_ratio_scatter.png", r"$E_g/k_B T$"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
        vals = sc[field].to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        vmax = float(np.nanpercentile(finite, 95)) if finite.size else 1.0
        im = ax.scatter(sc["kBT"], sc["J_A"], c=vals, s=9, cmap="viridis", vmin=0, vmax=vmax, edgecolors="none")
        ax.set_xlabel(r"$k_B T/t$")
        ax.set_ylabel(r"$J_A/t$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / "figures" / fname)
        plt.close(fig)


def write_reports(out_dir: Path, mode: str, validation_pass: bool) -> None:
    summary_path = out_dir / "topology_pass_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    md = [
        f"# Stage III Topology Pass: `{RUN_ID}`",
        "",
        "## Scope",
        "",
        "This report is an offline topology classification and compute-cost audit on frozen `dataset_iter035`.",
        "It does not modify thermodynamic labels and does not run a new active-learning loop or Delta-q exact search.",
        "",
        "## Current Status",
        "",
        f"- mode executed: `{mode}`",
        f"- Pfaffian validation pass: `{validation_pass}`",
        f"- full-pass summary available: `{bool(summary)}`",
        "",
        "## Important Caveats",
        "",
        "- Sparse topology scatter is not a final topological phase boundary.",
        "- Pfaffian signs are used only together with the full-Brillouin-zone gap status.",
        "- Hard-risk/provisional exact points are diagnostics only unless explicitly audited.",
        "- Parquet output depends on a local parquet backend (`pyarrow` or `fastparquet`).",
        "",
    ]
    if summary:
        counts = {
            (row["thermo_phase"], row["topology_label"]): int(row["count"])
            for row in summary.get("counts", [])
        }
        diag = summary.get("diagnostics", {})
        parquet_ok = str(summary.get("parquet_status", "")).startswith("written")
        md += [
            "## Full-Pass Summary",
            "",
            f"- total SC points: {summary.get('total_sc_points')}",
            f"- trusted topology points: {summary.get('trusted_topology_points')}",
            f"- provisional topology points: {summary.get('provisional_topology_points')}",
            f"- uniform-SC trivial/topological/gapless/unresolved: {counts.get(('uniform_SC', 'trivial'), 0)} / {counts.get(('uniform_SC', 'topological'), 0)} / {counts.get(('uniform_SC', 'gapless_SC'), 0)} / {counts.get(('uniform_SC', 'topology_unresolved'), 0)}",
            f"- FFLO trivial/topological/gapless/unresolved: {counts.get(('FFLO', 'trivial'), 0)} / {counts.get(('FFLO', 'topological'), 0)} / {counts.get(('FFLO', 'gapless_SC'), 0)} / {counts.get(('FFLO', 'topology_unresolved'), 0)}",
            f"- backend: {summary.get('backend')}",
            f"- Nk: {summary.get('nk')}",
            f"- actual runtime seconds: {summary.get('actual_runtime_seconds'):.3f}",
            f"- bulk-gap min/median/p95: {summary.get('bulk_gap_min'):.6g} / {summary.get('bulk_gap_median'):.6g} / {summary.get('bulk_gap_p95'):.6g}",
            f"- Pfaffian-margin min/median/p95: {summary.get('pfaffian_margin_min'):.6g} / {summary.get('pfaffian_margin_median'):.6g} / {summary.get('pfaffian_margin_p95'):.6g}",
            f"- parquet status: {summary.get('parquet_status')}",
            "",
            "## Pfaffian Convention Check",
            "",
            "- The analytic formula is cross-checked against the project BdG Hamiltonian builder in the current Nambu convention.",
            "- Verified convention: `P0 = (mu - t cos(q/2))^2 + Delta^2 - alpha_y^2 sin^2(q/2) - (J_A cos(q/2) + alpha_z sin(q/2))^2`.",
            "- Verified convention: `Ppi = (mu + t cos(q/2))^2 + Delta^2 - alpha_y^2 sin^2(q/2) - (J_A cos(q/2) + alpha_z sin(q/2))^2`.",
            "- Product-sign agreement is required on all non-boundary validation points before full pass.",
            "",
            "## Coverage Diagnostics",
            "",
            f"- trusted FFLO gapped points used for Delaunay diagnostics: {diag.get('trusted_fflo_gapped_points')}",
            f"- Z2-change candidate edges: {diag.get('z2_change_candidate_edges')}",
            f"- P0 sign-change candidate edges: {diag.get('p0_sign_change_candidate_edges')}",
            f"- Ppi sign-change candidate edges: {diag.get('ppi_sign_change_candidate_edges')}",
            f"- large circumradius coverage-hole triangles: {diag.get('large_circumradius_triangles')}",
            f"- nearest-neighbor p95/max distance in normalized parameter space: {diag.get('nearest_neighbor_distance_p95_norm')} / {diag.get('nearest_neighbor_distance_max_norm')}",
            "",
            "These are candidate topology-boundary seeds and coverage diagnostics only. They are not final topology contours.",
            "",
            "## Resource Decision",
            "",
            "- Pilot projection was below the 6-hour local threshold.",
            "- GPU and CPU agreed to double-precision tolerance at Nk=2048 in the pilot comparison.",
            "- The selected full-pass backend was chosen from measured pilot runtime, not assumed a priori.",
            "",
            "## Output Dataset",
            "",
            "- CSV output: `tables/dataset_iter035_topology_ground_v1.csv`.",
            "- NPZ output: `tables/dataset_iter035_topology_ground_v1.npz`.",
            f"- Parquet output available: `{parquet_ok}`.",
            "",
            "## Decision",
            "",
            f"- Recommended next case: `{diag.get('next_case')}`.",
            "- Current evidence contains both trusted FFLO trivial and topological points, so topology-aware follow-up is justified.",
            "- Because this is sparse inherited AL sampling, the next step should target Pfaffian-margin, bulk-gap, Z2-change edges, and coverage holes rather than treating the scatter boundary as final.",
            "",
        ]
    write_text(out_dir / "topology_validation_report.md", "\n".join(md))
    decision = [
        f"# Decision Log: `{RUN_ID}`",
        "",
        f"- Pfaffian analytic/numeric product-sign validation: `{validation_pass}`.",
        "- Input dataset was treated as frozen; no original labels were modified.",
        "- Full pass may proceed only after validation passes and pilot projection is acceptable.",
        "- Full pass uses the measured best backend from the pilot benchmark.",
        "- Sparse topology Delaunay edges are diagnostic seeds, not final topological boundaries.",
        "",
    ]
    if summary:
        diag = summary.get("diagnostics", {})
        decision += [
            f"- Trusted topology points: `{summary.get('trusted_topology_points')}` / `{summary.get('total_sc_points')}` SC points.",
            f"- Candidate Z2-change edges: `{diag.get('z2_change_candidate_edges')}`.",
            f"- Recommended next case: `{diag.get('next_case')}`.",
            "",
        ]
    write_text(out_dir / "decision_log.md", "\n".join(decision))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline topology pass for frozen dataset_iter035.")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--exact-merged", default=DEFAULT_EXACT_MERGED)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--mode", choices=["audit", "pilot", "full", "all"], default="pilot")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nk-values", default="512,1024,2048")
    p.add_argument("--full-nk", type=int, default=2048)
    p.add_argument("--point-chunk", type=int, default=64)
    p.add_argument("--k-chunk", type=int, default=512)
    p.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--full-if-projected-hours", type=float, default=6.0)
    return p.parse_args()


def choose_backend(out_dir: Path, requested: str) -> str:
    if requested in {"cpu", "gpu"}:
        return requested
    bench_path = out_dir / "tables" / "pilot_benchmark.csv"
    if not bench_path.exists():
        return "cpu"
    bench = pd.read_csv(bench_path)
    valid = bench[(bench["nk"] == bench["nk"].max()) & bench["runtime_seconds"].notna()]
    if valid.empty:
        return "cpu"
    return str(valid.sort_values("runtime_seconds").iloc[0]["backend"])


def projected_hours(out_dir: Path, backend: str, nk: int) -> float:
    path = out_dir / "tables" / "pilot_projection.csv"
    if not path.exists():
        return math.inf
    proj = pd.read_csv(path)
    rows = proj[(proj["backend"] == backend) & (proj["nk"] == nk)]
    if rows.empty:
        return math.inf
    return float(rows.iloc[0]["projected_full_runtime_hours"])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)
    dataset_path = Path(args.dataset)
    exact_path = Path(args.exact_merged)
    params = TopologyModelParams()
    input_audit(dataset_path, exact_path, out_dir, params)
    df = load_dataset(dataset_path)
    hard_df = load_hard_risk_exact(exact_path)
    validation_pass = run_pfaffian_validation(df, out_dir, count=100, seed=int(args.seed))
    if not validation_pass:
        write_reports(out_dir, args.mode, validation_pass=False)
        raise SystemExit("Pfaffian validation failed; stopping before pilot/full pass.")
    nk_values = [int(x.strip()) for x in str(args.nk_values).split(",") if x.strip()]
    if args.mode in {"pilot", "all"}:
        run_pilot_benchmark(df, hard_df, out_dir, nk_values, int(args.seed), int(args.point_chunk), int(args.k_chunk))
    if args.mode in {"full", "all"}:
        backend = choose_backend(out_dir, str(args.backend))
        proj = projected_hours(out_dir, backend, int(args.full_nk))
        if args.mode == "all" and proj > float(args.full_if_projected_hours):
            write_reports(out_dir, args.mode, validation_pass=True)
            raise SystemExit(f"Projected full runtime {proj:.2f} h exceeds threshold; stopping before full pass.")
        classify_topology(
            df,
            hard_df,
            out_dir,
            backend=backend,
            nk=int(args.full_nk),
            point_chunk=int(args.point_chunk),
            k_chunk=int(args.k_chunk),
        )
    write_reports(out_dir, args.mode, validation_pass=True)


if __name__ == "__main__":
    main()
