from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.stagev_acqv2 import (
    StageVConfig,
    build_boundary_support_sets,
    compute_point_rewards,
    generate_stagev_candidate_pool,
    score_stagev_a0,
    select_micro_batch,
    train_linear_value_model,
    update_lambda_t,
)
from ml_phase.dataset_builder import FlatDataset
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def synthetic_dataset(n: int, seed: int = 7) -> FlatDataset:
    rng = np.random.default_rng(seed)
    kT = rng.random(n) * 0.56
    mu = -0.5 + rng.random(n) * 2.0
    boundary = 0.35 + 0.7 * np.exp(-5.0 * kT) + 0.18 * np.sin(np.pi * mu)
    JA = rng.random(n) * 2.12
    sc = JA < boundary
    fflo = sc & (JA > 0.42 + 0.16 * np.cos(2.0 * np.pi * kT))
    phase = np.full(n, PHASE_NORMAL, dtype=np.int64)
    phase[sc] = PHASE_UNIFORM_SC
    phase[fflo] = PHASE_FFLO
    delta = np.where(sc, np.maximum(boundary - JA, 0.02), 0.0)
    q = np.where(fflo, 0.08 + 0.15 * np.abs(mu), 0.0)
    eta = np.zeros(n)
    y_reg = np.column_stack([delta, q, eta, np.zeros(n), np.zeros(n)])
    p0 = (mu - 0.4) * (JA - 0.55)
    ppi = mu + 2.0
    topo = np.full(n, -1, dtype=np.int64)
    topo[sc] = (p0[sc] < 0).astype(np.int64)
    records = {
        "kT": kT,
        "JA": JA,
        "mu": mu,
        "delta_opt": delta,
        "q_opt": q,
        "eta": eta,
        "ic_plus": np.zeros(n),
        "ic_minus": np.zeros(n),
        "phase_label": phase,
        "trusted_exact": np.ones(n, dtype=np.int8),
        "training_eligible_exact": np.ones(n, dtype=np.int8),
        "needs_rerun_exact": np.zeros(n, dtype=np.int8),
        "q_unresolved": np.zeros(n, dtype=np.int8),
        "delta_unresolved": np.zeros(n, dtype=np.int8),
        "free_energy_gap_to_normal": JA - boundary,
        "topology_trusted": sc.astype(np.int8),
        "topology_label_code": topo,
        "topology_spectral_status_code": np.where(sc, 0, -1),
        "topology_p0": p0,
        "topology_ppi": ppi,
        "topology_bulk_gap": np.where(sc, np.abs(p0) + 1.0e-3, np.nan),
    }
    return FlatDataset(
        x=np.column_stack([kT, JA, mu]),
        y_reg=y_reg.astype(np.float64),
        y_phase=phase,
        y_eta_sign=np.zeros(n, dtype=np.int64),
        y_strong_diode=np.zeros(n, dtype=np.int64),
        records=records,
    )


def synthetic_candidate_features(points: np.ndarray, cfg: StageVConfig) -> pd.DataFrame:
    kT, JA, mu = points[:, 0], points[:, 1], points[:, 2]
    ns_boundary = 0.35 + 0.7 * np.exp(-5.0 * kT) + 0.18 * np.sin(np.pi * mu)
    m_ns = JA - ns_boundary
    m_uf = JA - (0.42 + 0.16 * np.cos(2.0 * np.pi * kT))
    p0 = (mu - 0.4) * (JA - 0.55)
    ppi = mu + 2.0
    p_sc = 1.0 / (1.0 + np.exp(8.0 * m_ns))
    return pd.DataFrame(
        {
            "kT": kT,
            "JA": JA,
            "mu": mu,
            "p_normal": 1.0 - p_sc,
            "p_uniform_SC": p_sc * 0.35,
            "p_FFLO": p_sc * 0.65,
            "p_SC": p_sc,
            "pred_delta": np.maximum(-m_ns, 0.0),
            "pred_q": np.where(m_uf > 0, 0.12, 0.0),
            "m_NS": m_ns,
            "sigma_NS": np.full_like(m_ns, 0.03),
            "m_UF": m_uf,
            "sigma_UF": np.full_like(m_uf, 0.04),
            "m_P0": p0,
            "sigma_P0": np.full_like(p0, 0.04),
            "m_Ppi": ppi,
            "sigma_Ppi": np.full_like(ppi, 0.04),
            "m_gap": np.log(np.maximum(np.abs(p0), 1.0e-6) / cfg.gap_tol),
            "sigma_gap": np.full_like(p0, 0.2),
            "pf_product_pred": p0 * ppi,
            "pfaffian_margin_pred": np.minimum(np.abs(p0), np.abs(ppi)),
        }
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage V acquisition-v2 local smoke test.")
    p.add_argument("--output-dir", type=Path, default=Path("reports/stagev_acqv2_smoke"))
    p.add_argument("--seed", type=int, default=20260628)
    args = p.parse_args()
    out = args.output_dir
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    cfg = StageVConfig(candidate_pool_size=4096, micro_batch_size=64, reg_epochs=2, cls_epochs=2, model_ensemble=2, random_seed=int(args.seed))
    dataset = synthetic_dataset(640, seed=int(args.seed))
    support = build_boundary_support_sets(dataset, cfg)
    candidates, meta, cand_summary = generate_stagev_candidate_pool(dataset, cfg, iteration=1)
    features = synthetic_candidate_features(candidates, cfg)
    features["candidate_source"] = meta["candidate_source"].to_numpy()
    scored, score_summary = score_stagev_a0(features, support, cfg)
    selected, selected_meta, select_summary = select_micro_batch(scored, cfg, rng=np.random.default_rng(int(args.seed) + 1))
    rewards = compute_point_rewards(selected_meta)
    feature_cols = ["A0", "B_normal_sc", "B_p0_topology", "H_normal_sc", "H_p0_topology", "nearest_exact_distance"]
    model = train_linear_value_model(selected_meta.assign(**rewards), rewards["reward_scalar"].to_numpy(float), feature_cols)
    lambda_t = update_lambda_t(0.0, {"reward_sample_count": len(rewards), "rank_correlation_delta_vs_a0": 0.1}, cfg)
    scored.head(500).to_csv(tables / "stagev_smoke_candidate_scores_head.csv", index=False)
    selected_meta.to_csv(tables / "stagev_smoke_selected_points.csv", index=False)
    rewards.to_csv(tables / "stagev_smoke_rewards.csv", index=False)
    summary = {
        "status": "pass",
        "candidate_count": int(candidates.shape[0]),
        "selected_count": int(selected.shape[0]),
        "support_counts": {k: int(v.reshape(-1, 3).shape[0]) for k, v in support.items()},
        "candidate_summary": cand_summary,
        "score_summary": score_summary,
        "select_summary": select_summary,
        "reward_mean": float(rewards["reward_scalar"].mean()),
        "reward_model_status": model.get("status"),
        "lambda_t_after_smoke": float(lambda_t),
        "no_exact_calculation": True,
        "stageiv_data_used_for_training": False,
    }
    write_json(out / "stagev_acqv2_smoke_summary.json", summary)
    (out / "stagev_acqv2_smoke.md").write_text(
        "\n".join(
            [
                "# Stage V Acquisition-v2 Smoke Test",
                "",
                f"- status: `{summary['status']}`",
                f"- candidate_count: `{summary['candidate_count']}`",
                f"- selected_count: `{summary['selected_count']}`",
                f"- reward_model_status: `{summary['reward_model_status']}`",
                f"- no_exact_calculation: `{summary['no_exact_calculation']}`",
                f"- stageiv_data_used_for_training: `{summary['stageiv_data_used_for_training']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
