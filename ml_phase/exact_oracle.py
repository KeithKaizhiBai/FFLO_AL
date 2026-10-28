from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from eta_phase_diagram_cuda import (
    EtaPhaseConfig,
    build_q_vec,
    compute_current_from_omega,
    compute_omega_min_q_batch,
    find_eta_from_jq,
    maybe_set_linalg_backend,
)

from .topology_oracle import BulkGapOracle, TopologyModelParams, TopologyPfaffianOracle


PHASE_NORMAL = 0
PHASE_SUPERCONDUCTING = 1
PHASE_AMBIGUOUS = 2

Q_NOT_APPLICABLE = 0
Q_ACTIVE = 1
Q_EDGE_HIT = 2
Q_EXPANDED_CONFIRMED = 3
Q_UNRESOLVED = 4

DELTA_STABLE = 0
DELTA_BOUNDARY_AMBIGUOUS = 1
DELTA_REFINED_CONFIRMED = 2
DELTA_UNRESOLVED = 3

STATUS_Q_EDGE_UNRESOLVED = 1
STATUS_DELTA_BOUNDARY_UNRESOLVED = 2
STATUS_NONFINITE_OUTPUT = 4
STATUS_MAX_Q_REFINEMENT_REACHED = 8
STATUS_MAX_DELTA_REFINEMENT_REACHED = 16

PHASE_NAMES = {
    PHASE_NORMAL: "normal",
    PHASE_SUPERCONDUCTING: "superconducting",
    PHASE_AMBIGUOUS: "ambiguous",
}


def _phase_name(code: int) -> str:
    return PHASE_NAMES.get(int(code), "unknown")
Q_STATUS_NAMES = {
    Q_NOT_APPLICABLE: "not_applicable",
    Q_ACTIVE: "active",
    Q_EDGE_HIT: "edge_hit",
    Q_EXPANDED_CONFIRMED: "expanded_confirmed",
    Q_UNRESOLVED: "unresolved",
}
DELTA_STATUS_NAMES = {
    DELTA_STABLE: "stable",
    DELTA_BOUNDARY_AMBIGUOUS: "boundary_ambiguous",
    DELTA_REFINED_CONFIRMED: "refined_confirmed",
    DELTA_UNRESOLVED: "unresolved",
}


@dataclass
class PointOracleResult:
    kT: float
    JA: float
    eta: float
    q_opt: float
    delta_opt: float
    ic_plus: float
    ic_minus: float
    omega_global: float
    q_min: float
    q_max: float
    n_q: int
    q_index: int
    q_edge_distance: float
    q_edge_hit_raw: int
    delta_min: float
    delta_max: float
    n_delta: int


@dataclass
class ScanResult:
    q_vec: np.ndarray
    delta_star_q: np.ndarray
    deltaf_q: np.ndarray
    omega_sc_q: np.ndarray
    omega_normal_q: np.ndarray
    omega_normal_scalar: float
    q_opt: float
    delta_opt: float
    deltaf_min: float
    q_index: int
    dq: float
    q_min: float
    q_max: float
    n_q: int
    q_edge_margin: float
    qopt_edge_hit: int
    q_edge_distance: float
    scan_runtime_sec: float = 0.0
    sc_scan_runtime_sec: float = 0.0
    normal_scan_runtime_sec: float = 0.0
    q_points_evaluated: int = 0
    normal_q_points_evaluated: int = 0
    estimated_grid_evaluations: int = 0
    normal_scalar_reused: int = 0


@dataclass
class QScanCache:
    q_values: np.ndarray
    delta_star_q: np.ndarray
    deltaf_min_q: np.ndarray
    omega_min_q: np.ndarray
    omega_normal_scalar: float
    source_level: np.ndarray
    optional_metadata: dict[str, Any] | None = None


@dataclass
class ConfirmedPoint:
    kT: float
    JA: float
    eta: float
    q_opt: float
    delta_opt: float
    ic_plus: float
    ic_minus: float
    phase_candidate: int
    q_status: int
    q_min: float
    q_max: float
    n_q: int
    q_index: int
    q_edge_distance: float
    q_edge_hit: int
    q_refinement_level: int
    q_expanded: int
    q_unresolved: int
    delta_status: int
    delta_min: float
    delta_max: float
    n_delta: int
    n_delta_refined: int
    delta_refinement_level: int
    delta_boundary_ambiguous: int
    delta_refined: int
    delta_unresolved: int
    free_energy_gap_to_normal: float
    positive_delta_gap: float
    positive_delta_checked: int
    exact_status_code: int
    exact_status_name: str
    trusted_exact: int
    confidence_state: str = "trusted"
    training_eligible_exact: int = 1
    rerun_required: int = 0
    mu: float = 0.55
    oracle_mode: str = "legacy"
    search_mode: str = "legacy"
    initial_q_min: float = float("nan")
    initial_q_max: float = float("nan")
    final_q_min: float = float("nan")
    final_q_max: float = float("nan")
    initial_n_q: int = 0
    final_n_q: int = 0
    initial_dq: float = float("nan")
    final_dq: float = float("nan")
    q_expansion_count: int = 0
    q_expansion_directions: str = ""
    q_expansion_trigger: str = "none"
    q_window_coverage_valid: int = 0
    q_window_unresolved: int = 0
    qopt_edge_hit_initial: int = 0
    qopt_edge_hit_final: int = 0
    edge_risk_left_initial: int = 0
    edge_risk_right_initial: int = 0
    edge_risk_left_final: int = 0
    edge_risk_right_final: int = 0
    expanded_window_found_lower_branch: int = 0
    phase_changed_after_q_expansion: int = 0
    local_minima_count: int = 0
    refined_local_minima_count: int = 0
    near_degenerate_branch_count: int = 0
    selected_minimum_rank: int = 1
    branch_candidates_path: str = "N/A"
    delta_refinement_triggered: int = 0
    delta_refinement_valid: int = 0
    boundary_ambiguous: int = 0
    changed_after_delta_refinement: int = 0
    unresolved_reason: str = ""
    point_total_runtime_sec: float = 0.0
    base_scan_runtime_sec: float = 0.0
    q_expansion_runtime_sec: float = 0.0
    q_expansion_left_runtime_sec: float = 0.0
    q_expansion_right_runtime_sec: float = 0.0
    delta_refinement_runtime_sec: float = 0.0
    local_refinement_runtime_sec: float = 0.0
    merge_cache_runtime_sec: float = 0.0
    local_minima_detection_runtime_sec: float = 0.0
    fallback_full_rescan_runtime_sec: float = 0.0
    other_runtime_sec: float = 0.0
    base_q_points_evaluated: int = 0
    added_left_q_points_evaluated: int = 0
    added_right_q_points_evaluated: int = 0
    recomputed_q_points: int = 0
    total_q_points_evaluated: int = 0
    base_grid_evaluations: int = 0
    incremental_q_grid_evaluations: int = 0
    fallback_full_rescan_grid_evaluations: int = 0
    delta_refinement_grid_evaluations: int = 0
    local_refinement_grid_evaluations: int = 0
    total_estimated_grid_evaluations: int = 0
    incremental_expansion_used: int = 0
    fallback_full_rescan_used: int = 0
    fallback_full_rescan_reason: str = "N/A"
    local_minima_detected_count: int = 0
    clustered_basin_count: int = 0
    selected_refine_target_count: int = 0
    basin_clustering_enabled: int = 0
    basin_clustering_merged_count: int = 0
    energy_window_pruning_enabled: int = 0
    energy_window_pruned_count: int = 0
    local_boxes_refined_count: int = 0
    local_refinement_reused_count: int = 0
    topology_enabled: int = 0
    topology_applicable: int = 0
    topology_pending: int = 1
    topology_label_code: int = -1
    topology_z2: int = -1
    topology_spectral_status_code: int = -1
    topology_trusted: int = 0
    topology_p0: float = float("nan")
    topology_ppi: float = float("nan")
    topology_pf_product: float = float("nan")
    topology_pfaffian_margin: float = float("nan")
    topology_bulk_gap: float = float("nan")
    topology_k_at_bulk_gap: float = float("nan")
    topology_gap_tol: float = float("nan")
    topology_gap_nk: int = 0
    topology_gap_backend_code: int = -1
    topology_runtime_sec: float = 0.0
    topology_error_code: int = 0


@dataclass
class OracleResult:
    kT: np.ndarray
    JA: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    q_opt: np.ndarray
    delta_opt: np.ndarray
    ic_plus: np.ndarray
    ic_minus: np.ndarray
    phase_candidate: np.ndarray
    q_status: np.ndarray
    q_min: np.ndarray
    q_max: np.ndarray
    n_q: np.ndarray
    q_index: np.ndarray
    q_edge_distance: np.ndarray
    q_edge_hit: np.ndarray
    q_refinement_level: np.ndarray
    q_expanded: np.ndarray
    q_unresolved: np.ndarray
    delta_status: np.ndarray
    delta_min: np.ndarray
    delta_max: np.ndarray
    n_delta: np.ndarray
    n_delta_refined: np.ndarray
    delta_refinement_level: np.ndarray
    delta_boundary_ambiguous: np.ndarray
    delta_refined: np.ndarray
    delta_unresolved: np.ndarray
    free_energy_gap_to_normal: np.ndarray
    positive_delta_gap: np.ndarray
    positive_delta_checked: np.ndarray
    exact_status_code: np.ndarray
    exact_status_name: np.ndarray
    trusted_exact: np.ndarray
    confidence_state: np.ndarray
    training_eligible_exact: np.ndarray
    rerun_required: np.ndarray
    oracle_mode: np.ndarray
    search_mode: np.ndarray
    initial_q_min: np.ndarray
    initial_q_max: np.ndarray
    final_q_min: np.ndarray
    final_q_max: np.ndarray
    initial_n_q: np.ndarray
    final_n_q: np.ndarray
    initial_dq: np.ndarray
    final_dq: np.ndarray
    q_expansion_count: np.ndarray
    q_expansion_directions: np.ndarray
    q_expansion_trigger: np.ndarray
    q_window_coverage_valid: np.ndarray
    q_window_unresolved: np.ndarray
    qopt_edge_hit_initial: np.ndarray
    qopt_edge_hit_final: np.ndarray
    edge_risk_left_initial: np.ndarray
    edge_risk_right_initial: np.ndarray
    edge_risk_left_final: np.ndarray
    edge_risk_right_final: np.ndarray
    expanded_window_found_lower_branch: np.ndarray
    phase_changed_after_q_expansion: np.ndarray
    local_minima_count: np.ndarray
    refined_local_minima_count: np.ndarray
    near_degenerate_branch_count: np.ndarray
    selected_minimum_rank: np.ndarray
    branch_candidates_path: np.ndarray
    delta_refinement_triggered: np.ndarray
    delta_refinement_valid: np.ndarray
    boundary_ambiguous: np.ndarray
    changed_after_delta_refinement: np.ndarray
    unresolved_reason: np.ndarray
    point_total_runtime_sec: np.ndarray
    base_scan_runtime_sec: np.ndarray
    q_expansion_runtime_sec: np.ndarray
    q_expansion_left_runtime_sec: np.ndarray
    q_expansion_right_runtime_sec: np.ndarray
    delta_refinement_runtime_sec: np.ndarray
    local_refinement_runtime_sec: np.ndarray
    merge_cache_runtime_sec: np.ndarray
    local_minima_detection_runtime_sec: np.ndarray
    fallback_full_rescan_runtime_sec: np.ndarray
    other_runtime_sec: np.ndarray
    base_q_points_evaluated: np.ndarray
    added_left_q_points_evaluated: np.ndarray
    added_right_q_points_evaluated: np.ndarray
    recomputed_q_points: np.ndarray
    total_q_points_evaluated: np.ndarray
    base_grid_evaluations: np.ndarray
    incremental_q_grid_evaluations: np.ndarray
    fallback_full_rescan_grid_evaluations: np.ndarray
    delta_refinement_grid_evaluations: np.ndarray
    local_refinement_grid_evaluations: np.ndarray
    total_estimated_grid_evaluations: np.ndarray
    incremental_expansion_used: np.ndarray
    fallback_full_rescan_used: np.ndarray
    fallback_full_rescan_reason: np.ndarray
    local_minima_detected_count: np.ndarray
    clustered_basin_count: np.ndarray
    selected_refine_target_count: np.ndarray
    basin_clustering_enabled: np.ndarray
    basin_clustering_merged_count: np.ndarray
    energy_window_pruning_enabled: np.ndarray
    energy_window_pruned_count: np.ndarray
    local_boxes_refined_count: np.ndarray
    local_refinement_reused_count: np.ndarray
    topology_enabled: np.ndarray
    topology_applicable: np.ndarray
    topology_pending: np.ndarray
    topology_label_code: np.ndarray
    topology_z2: np.ndarray
    topology_spectral_status_code: np.ndarray
    topology_trusted: np.ndarray
    topology_p0: np.ndarray
    topology_ppi: np.ndarray
    topology_pf_product: np.ndarray
    topology_pfaffian_margin: np.ndarray
    topology_bulk_gap: np.ndarray
    topology_k_at_bulk_gap: np.ndarray
    topology_gap_tol: np.ndarray
    topology_gap_nk: np.ndarray
    topology_gap_backend_code: np.ndarray
    topology_runtime_sec: np.ndarray
    topology_error_code: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return asdict(self)


def _device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


TOPOLOGY_LABEL_NOT_APPLICABLE = -1
TOPOLOGY_LABEL_TRIVIAL = 0
TOPOLOGY_LABEL_TOPOLOGICAL = 1
TOPOLOGY_LABEL_GAPLESS_SC = 2
TOPOLOGY_LABEL_UNRESOLVED = 3

TOPOLOGY_SPECTRAL_NOT_APPLICABLE = -1
TOPOLOGY_SPECTRAL_GAPPED = 0
TOPOLOGY_SPECTRAL_GAPLESS = 1
TOPOLOGY_SPECTRAL_UNRESOLVED = 2


def _backend_code(backend: str) -> int:
    return 1 if str(backend).lower() == "gpu" else 0


def _attach_topology_diagnostics(
    point: ConfirmedPoint,
    enabled: bool,
    pf_oracle: TopologyPfaffianOracle | None,
    gap_oracle: BulkGapOracle | None,
    gap_nk: int,
    gap_tol_rel: float,
    gap_tol_abs: float,
    gap_backend: str,
) -> ConfirmedPoint:
    """Attach active-loop topology diagnostics without changing thermo labels.

    These fields are acquisition metadata.  Publication-grade topology labels
    remain the responsibility of the offline topology pass with its stricter
    convergence audit.
    """
    if not enabled:
        return point

    t0 = time.perf_counter()
    point.topology_enabled = 1
    point.topology_pending = 0
    point.topology_gap_nk = int(gap_nk)
    point.topology_gap_backend_code = _backend_code(gap_backend)

    if int(point.phase_candidate) != PHASE_SUPERCONDUCTING:
        point.topology_applicable = 0
        point.topology_label_code = TOPOLOGY_LABEL_NOT_APPLICABLE
        point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_NOT_APPLICABLE
        point.topology_runtime_sec = float(time.perf_counter() - t0)
        return point

    point.topology_applicable = 1
    if pf_oracle is None or gap_oracle is None:
        point.topology_label_code = TOPOLOGY_LABEL_UNRESOLVED
        point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_UNRESOLVED
        point.topology_error_code = 1
        point.topology_runtime_sec = float(time.perf_counter() - t0)
        return point

    try:
        p0, ppi, product, margin, z2 = pf_oracle.z2_status(
            np.asarray([point.delta_opt], dtype=np.float64),
            np.asarray([point.q_opt], dtype=np.float64),
            np.asarray([point.JA], dtype=np.float64),
            mu=np.asarray([point.mu], dtype=np.float64),
        )
        point.topology_p0 = float(p0[0])
        point.topology_ppi = float(ppi[0])
        point.topology_pf_product = float(product[0])
        point.topology_pfaffian_margin = float(margin[0])
        point.topology_z2 = int(z2[0])

        gap_res = gap_oracle.compute(
            np.asarray([point.delta_opt], dtype=np.float64),
            np.asarray([point.q_opt], dtype=np.float64),
            np.asarray([point.JA], dtype=np.float64),
            nk=int(gap_nk),
            mu=np.asarray([point.mu], dtype=np.float64),
        )
        gap = float(np.asarray(gap_res["bulk_gap"], dtype=np.float64)[0])
        k_at = float(np.asarray(gap_res["k_at_bulk_gap"], dtype=np.float64)[0])
        point.topology_bulk_gap = gap
        point.topology_k_at_bulk_gap = k_at

        e_scale = float(pf_oracle.params.energy_scale(point.JA, point.delta_opt, mu=point.mu))
        gap_tol = max(float(gap_tol_abs), float(gap_tol_rel) * max(e_scale, 1.0e-300))
        point.topology_gap_tol = gap_tol

        thermo_reliable = bool(int(point.training_eligible_exact)) and not bool(int(point.q_unresolved)) and not bool(int(point.delta_unresolved))
        if not np.isfinite(gap) or not np.isfinite(point.topology_pf_product):
            point.topology_label_code = TOPOLOGY_LABEL_UNRESOLVED
            point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_UNRESOLVED
            point.topology_error_code = 2
        elif gap <= gap_tol:
            point.topology_label_code = TOPOLOGY_LABEL_GAPLESS_SC
            point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_GAPLESS
            point.topology_trusted = int(thermo_reliable)
        elif int(point.topology_z2) == 0:
            point.topology_label_code = TOPOLOGY_LABEL_TRIVIAL
            point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_GAPPED
            point.topology_trusted = int(thermo_reliable)
        elif int(point.topology_z2) == 1:
            point.topology_label_code = TOPOLOGY_LABEL_TOPOLOGICAL
            point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_GAPPED
            point.topology_trusted = int(thermo_reliable)
        else:
            point.topology_label_code = TOPOLOGY_LABEL_UNRESOLVED
            point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_UNRESOLVED
            point.topology_error_code = 3
    except Exception:
        point.topology_label_code = TOPOLOGY_LABEL_UNRESOLVED
        point.topology_spectral_status_code = TOPOLOGY_SPECTRAL_UNRESOLVED
        point.topology_error_code = 4
    point.topology_runtime_sec = float(time.perf_counter() - t0)
    return point


def _q_edge_margin(q_vec: np.ndarray, q_edge_margin: float | None) -> float:
    if q_edge_margin is not None:
        return float(q_edge_margin)
    q_step = float(np.min(np.diff(q_vec))) if q_vec.size > 1 else float("inf")
    return max(2.0 * abs(q_step), 1e-12)


def evaluate_one_point_once(
    kT: float,
    JA: float,
    cfg: EtaPhaseConfig,
    device: torch.device,
    q_edge_margin: float | None = None,
) -> PointOracleResult:
    cfg_scaled = cfg.scaled()
    maybe_set_linalg_backend(cfg_scaled)

    q_vec = build_q_vec(cfg_scaled)
    q_vec_t = torch.as_tensor(q_vec, device=device, dtype=cfg_scaled.dtype)
    k_vec = torch.linspace(-math.pi, math.pi, cfg_scaled.n_k, dtype=cfg_scaled.dtype, device=device)

    kt_batch = torch.as_tensor([float(kT)], dtype=cfg_scaled.dtype, device=device)
    ja_batch = torch.as_tensor([float(JA)], dtype=cfg_scaled.dtype, device=device)
    omega_min_q_t, _delta_opt_q_t, q_opt_t, delta_opt_t = compute_omega_min_q_batch(
        kt_batch,
        ja_batch,
        cfg_scaled,
        k_vec,
        q_vec_t,
    )

    omega_min_q = omega_min_q_t.detach().cpu().numpy()
    q_opt = float(q_opt_t.detach().cpu().numpy()[0, 0])
    delta_opt = float(delta_opt_t.detach().cpu().numpy()[0, 0])
    omega_global = float(np.nanmin(omega_min_q[0, 0]))
    j_q = compute_current_from_omega(omega_min_q, q_vec)[0, 0]
    iq_opt = int(np.argmin(np.abs(q_vec - q_opt)))
    eta, ic_plus, ic_minus = find_eta_from_jq(j_q, q_vec, iq_opt)
    edge_distance = float(min(abs(q_opt - q_vec[0]), abs(q_vec[-1] - q_opt)))
    edge_hit = int(edge_distance <= _q_edge_margin(q_vec, q_edge_margin))

    return PointOracleResult(
        kT=float(kT),
        JA=float(JA),
        eta=float(eta),
        q_opt=float(q_opt),
        delta_opt=float(delta_opt),
        ic_plus=float(ic_plus),
        ic_minus=float(ic_minus),
        omega_global=float(omega_global),
        q_min=float(q_vec[0]),
        q_max=float(q_vec[-1]),
        n_q=int(q_vec.size),
        q_index=int(iq_opt),
        q_edge_distance=float(edge_distance),
        q_edge_hit_raw=int(edge_hit),
        delta_min=float(cfg_scaled.delta_min),
        delta_max=float(cfg_scaled.delta_max),
        n_delta=int(cfg_scaled.n_delta),
    )


def _run_scan_for_q_vec_with_normal(
    kT: float,
    JA: float,
    cfg: EtaPhaseConfig,
    device: torch.device,
    q_edge_margin: float | None,
    q_vec: np.ndarray,
    omega_normal_scalar: float | None = None,
) -> ScanResult:
    total_t0 = time.perf_counter()
    cfg_scaled = cfg.scaled()
    maybe_set_linalg_backend(cfg_scaled)

    q_vec = np.asarray(q_vec, dtype=np.float64)
    q_vec_t = torch.as_tensor(q_vec, device=device, dtype=cfg_scaled.dtype)
    k_vec = torch.linspace(-math.pi, math.pi, cfg_scaled.n_k, dtype=cfg_scaled.dtype, device=device)
    kt_batch = torch.as_tensor([float(kT)], dtype=cfg_scaled.dtype, device=device)
    ja_batch = torch.as_tensor([float(JA)], dtype=cfg_scaled.dtype, device=device)

    sc_t0 = time.perf_counter()
    omega_sc_t, delta_star_t, q_opt_t, delta_opt_t = compute_omega_min_q_batch(
        kt_batch,
        ja_batch,
        cfg_scaled,
        k_vec,
        q_vec_t,
    )
    sc_runtime = time.perf_counter() - sc_t0
    omega_sc_q = omega_sc_t.detach().cpu().numpy()[0, 0]
    delta_star_q = delta_star_t.detach().cpu().numpy()[0, 0]
    q_opt = float(q_opt_t.detach().cpu().numpy()[0, 0])
    delta_opt = float(delta_opt_t.detach().cpu().numpy()[0, 0])

    normal_runtime = 0.0
    normal_reused = int(omega_normal_scalar is not None and np.isfinite(float(omega_normal_scalar)))
    if normal_reused:
        omega_normal_scalar = float(omega_normal_scalar)
        omega_normal_q = np.full_like(omega_sc_q, float(omega_normal_scalar), dtype=np.float64)
        normal_q_points = 0
    else:
        normal_cfg = EtaPhaseConfig(
            q_min=float(q_vec[0]),
            q_max=float(q_vec[-1]),
            n_q=int(q_vec.size),
            delta_min=0.0,
            delta_max=0.0,
            n_delta=1,
            n_k=int(cfg.n_k),
        )
        normal_scaled = normal_cfg.scaled()
        normal_q_t = torch.as_tensor(q_vec, device=device, dtype=normal_scaled.dtype)
        normal_t0 = time.perf_counter()
        omega_n_t, _delta_n_t, _q_n_t, _d_n_t = compute_omega_min_q_batch(
            kt_batch,
            ja_batch,
            normal_scaled,
            k_vec,
            normal_q_t,
        )
        normal_runtime = time.perf_counter() - normal_t0
        omega_normal_q = omega_n_t.detach().cpu().numpy()[0, 0]
        omega_normal_scalar = float(np.nanmin(omega_normal_q))
        normal_q_points = int(q_vec.size)
    deltaf_q = np.asarray(omega_sc_q - omega_normal_scalar, dtype=np.float64)
    deltaf_min = float(np.nanmin(deltaf_q)) if np.isfinite(deltaf_q).any() else float("nan")
    q_index = int(np.nanargmin(np.abs(q_vec - q_opt))) if np.isfinite(q_opt) else -1
    dq = float(np.min(np.diff(q_vec))) if q_vec.size > 1 else float("nan")
    margin = _q_edge_margin(q_vec, q_edge_margin)
    q_edge_distance = float(min(abs(q_opt - q_vec[0]), abs(q_vec[-1] - q_opt))) if np.isfinite(q_opt) else float("nan")
    qopt_edge_hit = int(np.isfinite(q_edge_distance) and q_edge_distance <= margin)
    estimated_grid_evaluations = int(q_vec.size) * int(cfg_scaled.n_delta) + int(normal_q_points)
    return ScanResult(
        q_vec=np.asarray(q_vec, dtype=np.float64),
        delta_star_q=np.asarray(delta_star_q, dtype=np.float64),
        deltaf_q=np.asarray(deltaf_q, dtype=np.float64),
        omega_sc_q=np.asarray(omega_sc_q, dtype=np.float64),
        omega_normal_q=np.asarray(omega_normal_q, dtype=np.float64),
        omega_normal_scalar=float(omega_normal_scalar),
        q_opt=float(q_opt),
        delta_opt=float(delta_opt),
        deltaf_min=float(deltaf_min),
        q_index=int(q_index),
        dq=float(dq),
        q_min=float(q_vec[0]),
        q_max=float(q_vec[-1]),
        n_q=int(q_vec.size),
        q_edge_margin=float(margin),
        qopt_edge_hit=int(qopt_edge_hit),
        q_edge_distance=float(q_edge_distance),
        scan_runtime_sec=float(time.perf_counter() - total_t0),
        sc_scan_runtime_sec=float(sc_runtime),
        normal_scan_runtime_sec=float(normal_runtime),
        q_points_evaluated=int(q_vec.size),
        normal_q_points_evaluated=int(normal_q_points),
        estimated_grid_evaluations=int(estimated_grid_evaluations),
        normal_scalar_reused=int(normal_reused),
    )


def _run_scan_with_normal(
    kT: float,
    JA: float,
    cfg: EtaPhaseConfig,
    device: torch.device,
    q_edge_margin: float | None,
    omega_normal_scalar: float | None = None,
) -> ScanResult:
    cfg_scaled = cfg.scaled()
    q_vec = build_q_vec(cfg_scaled)
    return _run_scan_for_q_vec_with_normal(
        kT=kT,
        JA=JA,
        cfg=cfg,
        device=device,
        q_edge_margin=q_edge_margin,
        q_vec=q_vec,
        omega_normal_scalar=omega_normal_scalar,
    )


def _scan_to_cache(scan: ScanResult, source_level: int) -> QScanCache:
    return QScanCache(
        q_values=np.asarray(scan.q_vec, dtype=np.float64),
        delta_star_q=np.asarray(scan.delta_star_q, dtype=np.float64),
        deltaf_min_q=np.asarray(scan.deltaf_q, dtype=np.float64),
        omega_min_q=np.asarray(scan.omega_sc_q, dtype=np.float64),
        omega_normal_scalar=float(scan.omega_normal_scalar),
        source_level=np.full(int(scan.n_q), int(source_level), dtype=np.int64),
        optional_metadata={
            "q_min": float(scan.q_min),
            "q_max": float(scan.q_max),
            "dq": float(scan.dq),
            "n_q": int(scan.n_q),
        },
    )


def _merge_q_scan_caches(caches: list[QScanCache], atol: float = 1e-10) -> QScanCache:
    usable = [c for c in caches if np.asarray(c.q_values).size > 0]
    if not usable:
        raise ValueError("at least one non-empty QScanCache is required")
    q = np.concatenate([np.asarray(c.q_values, dtype=np.float64) for c in usable])
    d = np.concatenate([np.asarray(c.delta_star_q, dtype=np.float64) for c in usable])
    f = np.concatenate([np.asarray(c.deltaf_min_q, dtype=np.float64) for c in usable])
    o = np.concatenate([np.asarray(c.omega_min_q, dtype=np.float64) for c in usable])
    src = np.concatenate([np.asarray(c.source_level, dtype=np.int64) for c in usable])
    order = np.argsort(q)
    q = q[order]
    d = d[order]
    f = f[order]
    o = o[order]
    src = src[order]

    keep: list[int] = []
    for idx, q_i in enumerate(q):
        if keep and abs(float(q_i) - float(q[keep[-1]])) <= float(atol):
            prev = keep[-1]
            if np.isfinite(f[idx]) and (not np.isfinite(f[prev]) or f[idx] < f[prev]):
                keep[-1] = idx
            continue
        keep.append(idx)
    keep_arr = np.asarray(keep, dtype=np.int64)
    normal_scalar = float(usable[0].omega_normal_scalar)
    return QScanCache(
        q_values=q[keep_arr],
        delta_star_q=d[keep_arr],
        deltaf_min_q=f[keep_arr],
        omega_min_q=o[keep_arr],
        omega_normal_scalar=normal_scalar,
        source_level=src[keep_arr],
        optional_metadata={"merged_cache_count": len(usable)},
    )


def _cache_to_scan(cache: QScanCache, q_edge_margin: float | None) -> ScanResult:
    q = np.asarray(cache.q_values, dtype=np.float64)
    f = np.asarray(cache.deltaf_min_q, dtype=np.float64)
    d = np.asarray(cache.delta_star_q, dtype=np.float64)
    o = np.asarray(cache.omega_min_q, dtype=np.float64)
    if q.size == 0:
        raise ValueError("cannot convert empty QScanCache to ScanResult")
    idx = int(np.nanargmin(f)) if np.isfinite(f).any() else -1
    q_opt = float(q[idx]) if idx >= 0 else float("nan")
    delta_opt = float(d[idx]) if idx >= 0 else float("nan")
    deltaf_min = float(f[idx]) if idx >= 0 else float("nan")
    dq = float(np.min(np.diff(q))) if q.size > 1 else float("nan")
    margin = _q_edge_margin(q, q_edge_margin)
    q_edge_distance = float(min(abs(q_opt - q[0]), abs(q[-1] - q_opt))) if np.isfinite(q_opt) else float("nan")
    qopt_edge_hit = int(np.isfinite(q_edge_distance) and q_edge_distance <= margin)
    return ScanResult(
        q_vec=q,
        delta_star_q=d,
        deltaf_q=f,
        omega_sc_q=o,
        omega_normal_q=np.full_like(o, float(cache.omega_normal_scalar), dtype=np.float64),
        omega_normal_scalar=float(cache.omega_normal_scalar),
        q_opt=float(q_opt),
        delta_opt=float(delta_opt),
        deltaf_min=float(deltaf_min),
        q_index=int(idx),
        dq=float(dq),
        q_min=float(q[0]),
        q_max=float(q[-1]),
        n_q=int(q.size),
        q_edge_margin=float(margin),
        qopt_edge_hit=int(qopt_edge_hit),
        q_edge_distance=float(q_edge_distance),
        normal_scalar_reused=1,
    )


def _incremental_q_strips(old_scan: ScanResult, new_cfg: EtaPhaseConfig) -> tuple[np.ndarray, np.ndarray]:
    old_q = np.asarray(old_scan.q_vec, dtype=np.float64)
    if old_q.size < 2:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    dq = float(np.min(np.diff(old_q)))
    if not np.isfinite(dq) or dq <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    q_min_new = float(new_cfg.q_min)
    q_max_new = float(new_cfg.q_max)
    left: np.ndarray
    right: np.ndarray
    if q_min_new < float(old_q[0]) - 0.5 * dq:
        n_left = int(math.ceil((float(old_q[0]) - q_min_new) / dq))
        left = float(old_q[0]) - dq * np.arange(n_left, 0, -1, dtype=np.float64)
        left = left[left >= q_min_new - 1e-12]
    else:
        left = np.asarray([], dtype=np.float64)
    if q_max_new > float(old_q[-1]) + 0.5 * dq:
        n_right = int(math.ceil((q_max_new - float(old_q[-1])) / dq))
        right = float(old_q[-1]) + dq * np.arange(1, n_right + 1, dtype=np.float64)
        right = right[right <= q_max_new + 1e-12]
    else:
        right = np.asarray([], dtype=np.float64)
    return left.astype(np.float64), right.astype(np.float64)


def _local_minima_indices(y: np.ndarray) -> list[int]:
    idx: list[int] = []
    if y.size == 0:
        return idx
    for i in range(y.size):
        yi = y[i]
        if not np.isfinite(yi):
            continue
        left_ok = i == 0 or yi <= y[i - 1]
        right_ok = i == y.size - 1 or yi <= y[i + 1]
        if left_ok and right_ok:
            idx.append(i)
    if not idx and np.isfinite(y).any():
        idx = [int(np.nanargmin(y))]
    return sorted(set(idx), key=lambda i: float(y[i]))


def _diagnose_q_window(
    scan: ScanResult,
    delta_eps: float,
    ambiguity_tol: float,
    edge_band_steps: int = 3,
) -> dict[str, object]:
    q = scan.q_vec
    f = scan.deltaf_q
    d = scan.delta_star_q
    n = int(q.size)
    band = max(1, min(int(edge_band_steps), max(1, n // 4)))
    global_min = float(np.nanmin(f)) if np.isfinite(f).any() else float("nan")
    has_sc_trace = bool(np.isfinite(d).any() and np.nanmax(d) > float(delta_eps))
    has_condensation = bool(np.isfinite(global_min) and global_min < -float(ambiguity_tol))
    edge_sensitive = bool(has_sc_trace or has_condensation)
    left_band = slice(0, band)
    right_band = slice(max(0, n - band), n)
    left_min = float(np.nanmin(f[left_band])) if np.isfinite(f[left_band]).any() else float("nan")
    right_min = float(np.nanmin(f[right_band])) if np.isfinite(f[right_band]).any() else float("nan")
    left_low_energy = bool(edge_sensitive and np.isfinite(left_min) and np.isfinite(global_min) and left_min <= global_min + 5.0e-5)
    right_low_energy = bool(edge_sensitive and np.isfinite(right_min) and np.isfinite(global_min) and right_min <= global_min + 5.0e-5)
    left_descent_outward = bool(edge_sensitive and n >= 3 and np.isfinite(f[1]) and np.isfinite(f[0]) and (f[1] - f[0]) > 0)
    right_descent_outward = bool(edge_sensitive and n >= 3 and np.isfinite(f[-2]) and np.isfinite(f[-1]) and (f[-2] - f[-1]) > 0)
    left_pos_delta = bool(np.isfinite(d[left_band]).any() and np.nanmax(d[left_band]) > float(delta_eps))
    right_pos_delta = bool(np.isfinite(d[right_band]).any() and np.nanmax(d[right_band]) > float(delta_eps))
    left_near0 = bool(np.isfinite(f[left_band]).any() and np.nanmin(np.abs(f[left_band])) <= float(ambiguity_tol))
    right_near0 = bool(np.isfinite(f[right_band]).any() and np.nanmin(np.abs(f[right_band])) <= float(ambiguity_tol))
    minima_idx = _local_minima_indices(f)
    low_edge_left = 0
    low_edge_right = 0
    for i in minima_idx:
        if not np.isfinite(f[i]) or not np.isfinite(global_min):
            continue
        if (not edge_sensitive) or f[i] > global_min + 5.0e-5:
            continue
        if i < band:
            low_edge_left += 1
        if i >= n - band:
            low_edge_right += 1
    no_reliable_q_trace_for_old_normal = bool(scan.delta_opt < float(delta_eps) and (left_pos_delta or right_pos_delta or left_near0 or right_near0))
    window_brackets_low_energy_branch = bool(not left_low_energy and not right_low_energy and not low_edge_left and not low_edge_right and not scan.qopt_edge_hit)
    qopt_left = bool(np.isfinite(scan.q_opt) and (scan.q_opt - scan.q_min) <= scan.q_edge_margin)
    qopt_right = bool(np.isfinite(scan.q_opt) and (scan.q_max - scan.q_opt) <= scan.q_edge_margin)
    return {
        "qopt_edge_hit": bool(scan.qopt_edge_hit),
        "qopt_edge_margin": float(scan.q_edge_distance),
        "qopt_left": qopt_left,
        "qopt_right": qopt_right,
        "left_edge_low_energy": left_low_energy,
        "right_edge_low_energy": right_low_energy,
        "left_edge_descent_outward": left_descent_outward,
        "right_edge_descent_outward": right_descent_outward,
        "left_positive_delta_trace": left_pos_delta,
        "right_positive_delta_trace": right_pos_delta,
        "low_energy_local_minima_near_left_edge": int(low_edge_left),
        "low_energy_local_minima_near_right_edge": int(low_edge_right),
        "no_reliable_q_trace_for_old_normal": no_reliable_q_trace_for_old_normal,
        "window_brackets_low_energy_branch": window_brackets_low_energy_branch,
    }


def _select_expansion_direction(diag: dict[str, object], allow_symmetric_once: bool) -> tuple[str, str]:
    left = False
    right = False
    triggers: list[str] = []
    if bool(diag.get("qopt_left")):
        left = True
        triggers.append("qopt_edge_left")
    if bool(diag.get("qopt_right")):
        right = True
        triggers.append("qopt_edge_right")
    if bool(diag.get("left_edge_low_energy")) or bool(diag.get("left_edge_descent_outward")) or int(diag.get("low_energy_local_minima_near_left_edge", 0)) > 0:
        left = True
        triggers.append("left_low_energy")
    if bool(diag.get("right_edge_low_energy")) or bool(diag.get("right_edge_descent_outward")) or int(diag.get("low_energy_local_minima_near_right_edge", 0)) > 0:
        right = True
        triggers.append("right_low_energy")
    if bool(diag.get("no_reliable_q_trace_for_old_normal")):
        if bool(diag.get("left_positive_delta_trace")) or bool(diag.get("left_edge_low_energy")):
            left = True
            triggers.append("old_normal_left_trace")
        if bool(diag.get("right_positive_delta_trace")) or bool(diag.get("right_edge_low_energy")):
            right = True
            triggers.append("old_normal_right_trace")
    if not left and not right and not bool(diag.get("window_brackets_low_energy_branch")) and allow_symmetric_once:
        left = True
        right = True
        triggers.append("symmetric_unbracketed")
    if left and right:
        return "both", ";".join(sorted(set(triggers))) if triggers else "both"
    if left:
        return "left", ";".join(sorted(set(triggers))) if triggers else "left"
    if right:
        return "right", ";".join(sorted(set(triggers))) if triggers else "right"
    return "none", "none"


def _expand_cfg_keep_density(
    cfg: EtaPhaseConfig,
    direction: str,
    q_max_abs: float,
    side_pad_factor: float = 0.5,
) -> EtaPhaseConfig:
    width = float(cfg.q_max - cfg.q_min)
    if width <= 0:
        return cfg
    old_dq = width / max(1, int(cfg.n_q) - 1)
    pad = max(abs(old_dq) * 4.0, width * float(side_pad_factor))
    q_min_new = float(cfg.q_min)
    q_max_new = float(cfg.q_max)
    if direction in {"left", "both"}:
        q_min_new = max(-abs(float(q_max_abs)), q_min_new - pad)
    if direction in {"right", "both"}:
        q_max_new = min(abs(float(q_max_abs)), q_max_new + pad)
    if q_max_new <= q_min_new:
        return cfg
    new_width = q_max_new - q_min_new
    n_q_new = max(int(cfg.n_q), int(math.ceil(new_width / max(abs(old_dq), 1e-12))) + 1)
    return replace(cfg, q_min=float(q_min_new), q_max=float(q_max_new), n_q=int(n_q_new))


def _build_branch_candidates(
    scan: ScanResult,
    local_refine_energy_window: float,
    edge_band_steps: int = 3,
) -> list[dict[str, object]]:
    minima_idx = _local_minima_indices(scan.deltaf_q)
    if not minima_idx:
        return []
    global_min = float(np.nanmin(scan.deltaf_q)) if np.isfinite(scan.deltaf_q).any() else float("nan")
    n = int(scan.q_vec.size)
    band = max(1, min(int(edge_band_steps), max(1, n // 4)))
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(minima_idx, start=1):
        f_i = float(scan.deltaf_q[idx])
        e_above = float(f_i - global_min) if np.isfinite(global_min) else float("nan")
        dist_edge = float(min(abs(scan.q_vec[idx] - scan.q_min), abs(scan.q_max - scan.q_vec[idx])))
        rows.append(
            {
                "minimum_rank": int(rank),
                "grid_index": int(idx),
                "q_local_min": float(scan.q_vec[idx]),
                "Delta_local_min": float(scan.delta_star_q[idx]),
                "DeltaF_local_min": float(f_i),
                "energy_above_global": float(e_above),
                "distance_to_q_edge": float(dist_edge),
                "within_low_energy_window": bool(np.isfinite(e_above) and e_above <= float(local_refine_energy_window)),
                "edge_risk": bool(idx < band or idx >= n - band),
            }
        )
    return rows


def _truthy_flag(row: dict[str, object], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _delta_near_epsilon(row: dict[str, object], delta_eps: float, delta_refine_half_width: float) -> bool:
    delta_local = float(row.get("Delta_local_min", np.nan))
    return bool(np.isfinite(delta_local) and abs(delta_local - float(delta_eps)) <= float(delta_refine_half_width))


def _has_basin_risk_annotation(row: dict[str, object]) -> bool:
    return any(
        key in row
        for key in (
            "basin_has_global_best",
            "basin_has_edge_risk",
            "basin_has_delta_near_epsilon",
            "basin_has_near_degenerate",
            "basin_has_low_energy_window",
        )
    )


def _mandatory_basin_reasons(row: dict[str, object], delta_eps: float, delta_refine_half_width: float) -> list[str]:
    reasons: list[str] = []
    if _has_basin_risk_annotation(row):
        if _truthy_flag(row, "basin_has_global_best"):
            reasons.append("global_best")
        if _truthy_flag(row, "basin_has_edge_risk"):
            reasons.append("edge_risk")
        if _truthy_flag(row, "basin_has_delta_near_epsilon"):
            reasons.append("Delta_near_epsilon")
        if _truthy_flag(row, "basin_has_near_degenerate"):
            reasons.append("near_degenerate")
        return reasons
    if int(row.get("minimum_rank", 0)) == 1:
        reasons.append("global_best")
    if bool(row.get("edge_risk", False)):
        reasons.append("edge_risk")
    if _delta_near_epsilon(row, delta_eps, delta_refine_half_width):
        reasons.append("Delta_near_epsilon")
    if int(row.get("minimum_rank", 0)) != 1 and bool(row.get("within_low_energy_window", False)):
        reasons.append("near_degenerate")
    return reasons


def annotate_basin_risk_flags(
    row: dict[str, object],
    members: list[dict[str, object]],
    delta_eps: float,
    delta_refine_half_width: float,
) -> dict[str, object]:
    """Attach basin-level risk flags derived from all clustered members."""
    out = dict(row)
    member_rows = [dict(m) for m in members] if members else [dict(row)]
    global_best_count = sum(1 for member in member_rows if int(member.get("minimum_rank", 0)) == 1)
    edge_risk_count = sum(1 for member in member_rows if bool(member.get("edge_risk", False)))
    delta_near_count = sum(1 for member in member_rows if _delta_near_epsilon(member, delta_eps, delta_refine_half_width))
    low_energy_count = sum(1 for member in member_rows if bool(member.get("within_low_energy_window", False)))
    near_degenerate_count = sum(
        1
        for member in member_rows
        if int(member.get("minimum_rank", 0)) != 1 and bool(member.get("within_low_energy_window", False))
    )
    flags: list[str] = []
    if global_best_count:
        flags.append("global_best")
    if edge_risk_count:
        flags.append("edge_risk")
    if delta_near_count:
        flags.append("Delta_near_epsilon")
    if near_degenerate_count:
        flags.append("near_degenerate")
    if low_energy_count:
        flags.append("low_energy_window")
    mandatory_flags = [flag for flag in flags if flag != "low_energy_window"]
    if not mandatory_flags:
        flags.append("ordinary")

    out["basin_has_global_best"] = bool(global_best_count)
    out["basin_has_edge_risk"] = bool(edge_risk_count)
    out["basin_has_delta_near_epsilon"] = bool(delta_near_count)
    out["basin_has_near_degenerate"] = bool(near_degenerate_count)
    out["basin_has_low_energy_window"] = bool(low_energy_count)
    out["basin_is_ordinary"] = bool(not mandatory_flags)
    out["basin_global_best_candidate_count"] = int(global_best_count)
    out["basin_edge_risk_candidate_count"] = int(edge_risk_count)
    out["basin_delta_near_epsilon_candidate_count"] = int(delta_near_count)
    out["basin_near_degenerate_candidate_count"] = int(near_degenerate_count)
    out["basin_low_energy_window_candidate_count"] = int(low_energy_count)
    out["basin_risk_flags"] = ";".join(flags)
    out["mandatory_basin_reasons"] = ";".join(mandatory_flags) if mandatory_flags else "none"
    out["mandatory_basin"] = bool(mandatory_flags)
    return out


def _candidate_same_basin(
    row: dict[str, object],
    representative: dict[str, object],
    q_tol: float,
    delta_tol: float,
    energy_tol: float,
) -> bool:
    q_a = float(row.get("q_local_min", np.nan))
    q_b = float(representative.get("q_local_min", np.nan))
    d_a = float(row.get("Delta_local_min", np.nan))
    d_b = float(representative.get("Delta_local_min", np.nan))
    f_a = float(row.get("DeltaF_local_min", np.nan))
    f_b = float(representative.get("DeltaF_local_min", np.nan))
    if not np.isfinite([q_a, q_b, d_a, d_b, f_a, f_b]).all():
        return False
    return bool(abs(q_a - q_b) <= float(q_tol) and abs(d_a - d_b) <= float(delta_tol) and abs(f_a - f_b) <= float(energy_tol))


def estimate_basin_geometry(rows: list[dict[str, object]]) -> dict[str, float]:
    q_values = np.asarray([float(r.get("q_local_min", np.nan)) for r in rows], dtype=np.float64)
    delta_values = np.asarray([float(r.get("Delta_local_min", np.nan)) for r in rows], dtype=np.float64)
    energy_values = np.asarray([float(r.get("DeltaF_local_min", np.nan)) for r in rows], dtype=np.float64)
    q_finite = q_values[np.isfinite(q_values)]
    delta_finite = delta_values[np.isfinite(delta_values)]
    energy_finite = energy_values[np.isfinite(energy_values)]
    q_width = float(np.max(q_finite) - np.min(q_finite)) if q_finite.size else 0.0
    delta_width = float(np.max(delta_finite) - np.min(delta_finite)) if delta_finite.size else 0.0
    energy_span = float(np.max(energy_finite) - np.min(energy_finite)) if energy_finite.size else 0.0
    denominator = max(q_width * q_width + delta_width * delta_width, 1.0e-24)
    curvature_proxy = float(energy_span / denominator)
    return {
        "basin_q_width": q_width,
        "basin_Delta_width": delta_width,
        "basin_energy_span": energy_span,
        "basin_curvature_proxy": curvature_proxy,
    }


def adaptive_local_box_half_widths(
    row: dict[str, object],
    default_q_half_width: float,
    default_delta_half_width: float,
    enabled: bool = False,
    min_factor: float = 0.5,
    max_factor: float = 2.0,
) -> tuple[float, float]:
    if not enabled:
        return float(default_q_half_width), float(default_delta_half_width)
    q_width = max(0.0, float(row.get("basin_q_width", 0.0)))
    delta_width = max(0.0, float(row.get("basin_Delta_width", 0.0)))
    q_min = float(default_q_half_width) * float(min_factor)
    q_max = float(default_q_half_width) * float(max_factor)
    d_min = float(default_delta_half_width) * float(min_factor)
    d_max = float(default_delta_half_width) * float(max_factor)
    q_half = min(q_max, max(q_min, max(float(default_q_half_width), 2.0 * q_width)))
    d_half = min(d_max, max(d_min, max(float(default_delta_half_width), 2.0 * delta_width)))
    return float(q_half), float(d_half)


BRANCH_REUSE_CACHE_FIELDS = [
    "point_id",
    "branch_id",
    "basin_id",
    "cluster_id",
    "q_refined",
    "Delta_refined",
    "DeltaF_refined",
    "q_window_level",
    "solver_config_signature",
    "local_box_signature",
    "reuse_cache_valid",
]

BRANCH_REUSE_DIAGNOSTIC_FIELDS = [
    "branch_id",
    "basin_id",
    "reuse_attempted",
    "reuse_allowed",
    "reuse_rejection_reason",
    "solver_config_signature",
    "local_box_signature",
    "cached_solver_config_signature",
    "cached_local_box_signature",
    "candidate_q",
    "candidate_Delta",
    "candidate_DeltaF",
    "cached_q",
    "cached_Delta",
    "cached_DeltaF",
    "q_abs_diff",
    "Delta_abs_diff",
    "energy_abs_diff",
    "lower_competing_deltaf",
]

ADAPTIVE_LOCAL_BOX_DIAGNOSTIC_FIELDS = [
    "adaptive_box_enabled",
    "default_q_half_width",
    "default_delta_half_width",
    "suggested_q_half_width",
    "suggested_delta_half_width",
    "min_factor",
    "max_factor",
    "basin_q_width",
    "basin_Delta_width",
    "basin_energy_span",
    "basin_curvature_proxy",
    "adaptive_box_reason",
]

LOCAL_BOX_BATCH_PLAN_FIELDS = [
    "batching_enabled",
    "batch_id",
    "box_count",
    "point_id",
    "kT",
    "JA",
    "q_window_level",
    "branch_ids",
    "grid_shape",
    "n_q_local_max",
    "n_Delta_local_max",
    "local_grid_evaluations",
    "box_q_min",
    "box_q_max",
    "box_Delta_min",
    "box_Delta_max",
    "dtype",
    "device",
    "batch_plan_reason",
]

HAMILTONIAN_CACHE_SIGNATURE_FIELDS = [
    "solver_mode",
    "code_version",
    "dtype",
    "device",
    "backend",
    "n_k",
    "model_signature",
    "kT",
    "JA",
    "q_grid_signature",
    "delta_grid_signature",
    "local_box_signature",
]

HAMILTONIAN_CACHE_DIAGNOSTIC_FIELDS = [
    "cache_enabled",
    "cache_lookup_attempted",
    "cache_hit_allowed",
    "cache_rejection_reason",
    "expected_cache_signature",
    "cached_cache_signature",
    "expected_tensor_shape",
    "cached_tensor_shape",
    "expected_dtype",
    "cached_dtype",
    "expected_device",
    "cached_device",
]

LOCAL_BOX_PROFILER_EVENT_FIELDS = [
    "profiler_scope",
    "event_name",
    "point_id",
    "branch_id",
    "batch_id",
    "batching_enabled",
    "cache_lookup_attempted",
    "cache_hit_allowed",
    "tensor_construction_location",
    "grid_shape",
    "runtime_sec",
    "local_grid_evaluations",
    "profiler_note",
]


def build_adaptive_local_box_diagnostic_record(
    row: dict[str, object],
    default_q_half_width: float,
    default_delta_half_width: float,
    enabled: bool = False,
    min_factor: float = 0.5,
    max_factor: float = 2.0,
) -> dict[str, object]:
    q_half, delta_half = adaptive_local_box_half_widths(
        row,
        default_q_half_width=default_q_half_width,
        default_delta_half_width=default_delta_half_width,
        enabled=enabled,
        min_factor=min_factor,
        max_factor=max_factor,
    )
    q_max = float(default_q_half_width) * float(max_factor)
    d_max = float(default_delta_half_width) * float(max_factor)
    if not enabled:
        reason = "adaptive_disabled_fixed_box"
    elif abs(q_half - q_max) <= 1.0e-15 or abs(delta_half - d_max) <= 1.0e-15:
        reason = "bounded_by_max_factor"
    elif abs(q_half - float(default_q_half_width)) <= 1.0e-15 and abs(delta_half - float(default_delta_half_width)) <= 1.0e-15:
        reason = "default_width_sufficient"
    else:
        reason = "adaptive_width_suggested"
    return {
        "adaptive_box_enabled": int(bool(enabled)),
        "default_q_half_width": float(default_q_half_width),
        "default_delta_half_width": float(default_delta_half_width),
        "suggested_q_half_width": float(q_half),
        "suggested_delta_half_width": float(delta_half),
        "min_factor": float(min_factor),
        "max_factor": float(max_factor),
        "basin_q_width": float(row.get("basin_q_width", 0.0)),
        "basin_Delta_width": float(row.get("basin_Delta_width", 0.0)),
        "basin_energy_span": float(row.get("basin_energy_span", 0.0)),
        "basin_curvature_proxy": float(row.get("basin_curvature_proxy", 0.0)),
        "adaptive_box_reason": reason,
    }


def _row_float(row: dict[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _row_int(row: dict[str, object], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _shared_float(rows: list[dict[str, object]], key: str) -> float:
    values = [_row_float(row, key) for row in rows]
    finite = [v for v in values if np.isfinite(v)]
    if len(finite) == len(values) and len({round(v, 15) for v in finite}) == 1:
        return float(finite[0])
    return float("nan")


def _shared_int(rows: list[dict[str, object]], key: str, default: int = -1) -> int:
    values = [_row_int(row, key, default=default) for row in rows]
    unique = set(values)
    if len(unique) == 1:
        return int(values[0])
    return int(default)


def _shape_string(shape: object) -> str:
    if shape is None:
        return ""
    if isinstance(shape, str):
        return shape
    try:
        return "x".join(str(int(v)) for v in shape)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(shape)


def build_local_box_batch_plan(
    rows: list[dict[str, object]],
    *,
    enabled: bool = False,
    max_boxes_per_batch: int = 1,
    dtype: str = "float64",
    device: str = "cuda",
) -> list[dict[str, object]]:
    """Build an auditable future batching plan without evaluating any boxes."""
    if not rows:
        return []
    if not enabled:
        chunk_size = 1
        reason = "batching_disabled_single_box"
    else:
        chunk_size = int(max_boxes_per_batch)
        if chunk_size <= 0:
            chunk_size = 1
            reason = "invalid_max_boxes_single_box"
        elif chunk_size == 1:
            reason = "enabled_single_box_batch"
        else:
            reason = "enabled_chunked_batch"

    plan: list[dict[str, object]] = []
    for batch_id, start in enumerate(range(0, len(rows), chunk_size)):
        chunk = [dict(row) for row in rows[start : start + chunk_size]]
        n_q_values = [max(0, _row_int(row, "n_q_local")) for row in chunk]
        n_delta_values = [max(0, _row_int(row, "n_Delta_local")) for row in chunk]
        n_q_max = max(n_q_values) if n_q_values else 0
        n_delta_max = max(n_delta_values) if n_delta_values else 0
        grid_evaluations = 0
        for row, n_q, n_delta in zip(chunk, n_q_values, n_delta_values):
            explicit_evals = _row_int(row, "local_grid_evaluations", default=-1)
            grid_evaluations += int(explicit_evals if explicit_evals >= 0 else n_q * n_delta)
        branch_ids = [
            str(_row_int(row, "branch_id", default=_row_int(row, "minimum_rank", default=-1)))
            for row in chunk
        ]
        q_mins = [_row_float(row, "box_q_min") for row in chunk]
        q_maxs = [_row_float(row, "box_q_max") for row in chunk]
        d_mins = [_row_float(row, "box_Delta_min") for row in chunk]
        d_maxs = [_row_float(row, "box_Delta_max") for row in chunk]
        q_mins_f = [v for v in q_mins if np.isfinite(v)]
        q_maxs_f = [v for v in q_maxs if np.isfinite(v)]
        d_mins_f = [v for v in d_mins if np.isfinite(v)]
        d_maxs_f = [v for v in d_maxs if np.isfinite(v)]
        plan.append(
            {
                "batching_enabled": int(bool(enabled)),
                "batch_id": int(batch_id),
                "box_count": int(len(chunk)),
                "point_id": int(_shared_int(chunk, "point_id", default=-1)),
                "kT": float(_shared_float(chunk, "kT")),
                "JA": float(_shared_float(chunk, "JA")),
                "q_window_level": int(_shared_int(chunk, "q_window_level", default=-1)),
                "branch_ids": ";".join(branch_ids),
                "grid_shape": f"boxes={len(chunk)};n_q_max={n_q_max};n_Delta_max={n_delta_max}",
                "n_q_local_max": int(n_q_max),
                "n_Delta_local_max": int(n_delta_max),
                "local_grid_evaluations": int(grid_evaluations),
                "box_q_min": float(min(q_mins_f)) if q_mins_f else float("nan"),
                "box_q_max": float(max(q_maxs_f)) if q_maxs_f else float("nan"),
                "box_Delta_min": float(min(d_mins_f)) if d_mins_f else float("nan"),
                "box_Delta_max": float(max(d_maxs_f)) if d_maxs_f else float("nan"),
                "dtype": str(dtype),
                "device": str(device),
                "batch_plan_reason": reason,
            }
        )
    return plan


def build_hamiltonian_cache_signature(values: dict[str, object]) -> str:
    payload = {field: values.get(field, "") for field in HAMILTONIAN_CACHE_SIGNATURE_FIELDS}
    return branch_reuse_signature(payload)


def evaluate_hamiltonian_cache_candidate(
    cached_entry: dict[str, object] | None,
    *,
    expected_cache_signature: str,
    expected_tensor_shape: object,
    expected_dtype: str,
    expected_device: str,
    cache_enabled: bool = False,
) -> dict[str, object]:
    """Decide whether a future Hamiltonian cache entry may be used."""
    expected_shape = _shape_string(expected_tensor_shape)
    if not cache_enabled:
        return {"cache_hit_allowed": False, "cache_rejection_reason": "cache_disabled"}
    if cached_entry is None:
        return {"cache_hit_allowed": False, "cache_rejection_reason": "missing_cached_entry"}
    cached_signature = str(cached_entry.get("hamiltonian_cache_signature", cached_entry.get("cache_signature", "")))
    if cached_signature != str(expected_cache_signature):
        return {"cache_hit_allowed": False, "cache_rejection_reason": "hamiltonian_cache_signature_mismatch"}
    cached_shape = _shape_string(cached_entry.get("tensor_shape", ""))
    if cached_shape != expected_shape:
        return {"cache_hit_allowed": False, "cache_rejection_reason": "tensor_shape_mismatch"}
    if str(cached_entry.get("dtype", "")) != str(expected_dtype):
        return {"cache_hit_allowed": False, "cache_rejection_reason": "dtype_mismatch"}
    if str(cached_entry.get("device", "")) != str(expected_device):
        return {"cache_hit_allowed": False, "cache_rejection_reason": "device_mismatch"}
    return {"cache_hit_allowed": True, "cache_rejection_reason": "cache_hit_allowed"}


def build_hamiltonian_cache_diagnostic_record(
    cached_entry: dict[str, object] | None,
    decision: dict[str, object],
    *,
    expected_cache_signature: str,
    expected_tensor_shape: object,
    expected_dtype: str,
    expected_device: str,
    cache_enabled: bool = False,
) -> dict[str, object]:
    expected_shape = _shape_string(expected_tensor_shape)
    return {
        "cache_enabled": int(bool(cache_enabled)),
        "cache_lookup_attempted": int(bool(cache_enabled)),
        "cache_hit_allowed": int(bool(decision.get("cache_hit_allowed", False))),
        "cache_rejection_reason": str(decision.get("cache_rejection_reason", "missing_cache_decision")),
        "expected_cache_signature": str(expected_cache_signature),
        "cached_cache_signature": (
            str(cached_entry.get("hamiltonian_cache_signature", cached_entry.get("cache_signature", "")))
            if cached_entry is not None
            else ""
        ),
        "expected_tensor_shape": expected_shape,
        "cached_tensor_shape": _shape_string(cached_entry.get("tensor_shape", "")) if cached_entry is not None else "",
        "expected_dtype": str(expected_dtype),
        "cached_dtype": str(cached_entry.get("dtype", "")) if cached_entry is not None else "",
        "expected_device": str(expected_device),
        "cached_device": str(cached_entry.get("device", "")) if cached_entry is not None else "",
    }


def build_local_box_profiler_event(
    *,
    profiler_scope: str,
    event_name: str,
    point_id: int = -1,
    branch_id: int = -1,
    batch_id: int = -1,
    batching_enabled: bool = False,
    cache_lookup_attempted: bool = False,
    cache_hit_allowed: bool = False,
    tensor_construction_location: str = "",
    grid_shape: object = "",
    runtime_sec: float = 0.0,
    local_grid_evaluations: int = 0,
    profiler_note: str = "",
) -> dict[str, object]:
    """Build a stable profiler event row for future GPU-level audits."""
    return {
        "profiler_scope": str(profiler_scope),
        "event_name": str(event_name),
        "point_id": int(point_id),
        "branch_id": int(branch_id),
        "batch_id": int(batch_id),
        "batching_enabled": int(bool(batching_enabled)),
        "cache_lookup_attempted": int(bool(cache_lookup_attempted)),
        "cache_hit_allowed": int(bool(cache_hit_allowed)),
        "tensor_construction_location": str(tensor_construction_location),
        "grid_shape": _shape_string(grid_shape),
        "runtime_sec": float(runtime_sec),
        "local_grid_evaluations": int(local_grid_evaluations),
        "profiler_note": str(profiler_note),
    }


def cluster_branch_candidates(
    rows: list[dict[str, object]],
    coarse_dq: float,
    coarse_dDelta: float,
    numerical_energy_scale: float,
    delta_eps: float,
    delta_refine_half_width: float,
    q_cluster_factor: float = 1.5,
    delta_cluster_factor: float = 1.5,
    energy_cluster_factor: float = 1.0,
) -> list[dict[str, object]]:
    """Cluster duplicate coarse minima while carrying mandatory-risk metadata."""
    if not rows:
        return []
    q_tol = max(0.0, abs(float(coarse_dq)) * float(q_cluster_factor))
    delta_tol = max(0.0, abs(float(coarse_dDelta)) * float(delta_cluster_factor))
    energy_tol = max(0.0, abs(float(numerical_energy_scale)) * float(energy_cluster_factor))
    clusters: list[dict[str, object]] = []
    for row in sorted((dict(r) for r in rows), key=lambda r: float(r.get("DeltaF_local_min", np.inf))):
        match: dict[str, object] | None = None
        for cluster in clusters:
            if _candidate_same_basin(row, cluster["representative"], q_tol, delta_tol, energy_tol):
                match = cluster
                break
        if match is None:
            clusters.append({"representative": dict(row), "members": [dict(row)]})
        else:
            match["members"].append(dict(row))

    representatives: list[dict[str, object]] = []
    for basin_id, cluster in enumerate(clusters, start=1):
        members = list(cluster["members"])
        rep = annotate_basin_risk_flags(
            dict(cluster["representative"]),
            [dict(member) for member in members],
            delta_eps=delta_eps,
            delta_refine_half_width=delta_refine_half_width,
        )
        member_ids = [int(m.get("minimum_rank", 0)) for m in members]
        rep["basin_id"] = int(basin_id)
        rep["cluster_size"] = int(len(members))
        rep["merged_branch_ids"] = ";".join(str(i) for i in sorted(member_ids))
        rep["cluster_reason"] = "q_delta_energy_duplicate" if len(members) > 1 else "singleton"
        rep["edge_risk"] = bool(any(bool(m.get("edge_risk", False)) for m in members))
        rep["within_low_energy_window"] = bool(any(bool(m.get("within_low_energy_window", False)) for m in members))
        rep.update(estimate_basin_geometry(members))
        representatives.append(rep)
    return sorted(representatives, key=lambda r: float(r.get("DeltaF_local_min", np.inf)))


def select_local_refine_targets(
    rows: list[dict[str, object]],
    delta_eps: float,
    delta_refine_half_width: float,
    max_total_refined_basins: int,
    enable_selective_refinement: bool = False,
    max_optional_refined_basins: int = 3,
    mandatory_basins_can_exceed_cap: bool = True,
    high_risk_overflow_policy: str = "keep_all",
    max_edge_risk_basins: int = 1,
    max_delta_near_eps_basins: int = 2,
    max_near_degenerate_basins: int = 2,
) -> list[dict[str, object]]:
    """Select local-refinement targets, preserving legacy behavior by default."""
    sorted_rows = sorted((dict(r) for r in rows), key=lambda r: float(r.get("DeltaF_local_min", np.inf)))
    if not enable_selective_refinement:
        legacy_targets: list[dict[str, object]] = []
        for row in sorted_rows:
            if bool(row.get("pruned_by_energy_window", False)):
                continue
            if int(row["minimum_rank"]) == 1:
                legacy_targets.append(row)
                continue
            if (
                bool(row["within_low_energy_window"])
                or bool(row["edge_risk"])
                or abs(float(row["Delta_local_min"]) - float(delta_eps)) <= float(delta_refine_half_width)
            ):
                legacy_targets.append(row)
        if len(legacy_targets) > int(max_total_refined_basins):
            legacy_targets = legacy_targets[: int(max_total_refined_basins)]
        return legacy_targets

    mandatory: list[dict[str, object]] = []
    ordinary: list[dict[str, object]] = []
    for row in sorted_rows:
        if bool(row.get("pruned_by_energy_window", False)):
            continue
        reasons = _mandatory_basin_reasons(row, delta_eps, delta_refine_half_width)
        row["mandatory_basin_reasons"] = ";".join(sorted(reasons)) if reasons else str(row.get("mandatory_basin_reasons", "none"))
        row["mandatory_basin"] = bool(reasons)
        row["basin_selection_reason"] = _local_refine_selection_reason(row, delta_eps, delta_refine_half_width)
        if reasons:
            mandatory.append(row)
        else:
            ordinary.append(row)

    overflow_policy = str(high_risk_overflow_policy).strip().lower()
    if overflow_policy not in {"keep_all", "rank_and_cap"}:
        raise ValueError(f"Unsupported high_risk_overflow_policy={high_risk_overflow_policy!r}")

    if overflow_policy == "rank_and_cap":
        total_cap = max(0, int(max_total_refined_basins))
        selected: list[dict[str, object]] = []
        selected_keys: set[object] = set()

        def row_key(row: dict[str, object]) -> object:
            if "basin_id" in row:
                return ("basin", int(row.get("basin_id", 0)))
            return ("rank", int(row.get("minimum_rank", 0)))

        def append_ranked(candidates: list[dict[str, object]], limit: int, reason: str) -> None:
            nonlocal selected
            for row in candidates[: max(0, int(limit))]:
                if len(selected) >= total_cap:
                    break
                key = row_key(row)
                if key in selected_keys:
                    continue
                row["mandatory_overflow_policy"] = "rank_and_cap"
                row["mandatory_overflow"] = False
                row["mandatory_overflow_count"] = 0
                row["rank_and_cap_selection_reason"] = reason
                selected.append(row)
                selected_keys.add(key)

        def energy_key(row: dict[str, object]) -> tuple[float, float, int]:
            return (
                float(row.get("energy_above_global", np.inf)),
                float(row.get("DeltaF_local_min", np.inf)),
                int(row.get("minimum_rank", 0)),
            )

        def edge_key(row: dict[str, object]) -> tuple[float, float, float, int]:
            return (
                float(row.get("distance_to_q_edge", np.inf)),
                float(row.get("energy_above_global", np.inf)),
                float(row.get("DeltaF_local_min", np.inf)),
                int(row.get("minimum_rank", 0)),
            )

        def delta_key(row: dict[str, object]) -> tuple[float, float, float, int]:
            return (
                abs(float(row.get("Delta_local_min", np.inf)) - float(delta_eps)),
                float(row.get("energy_above_global", np.inf)),
                float(row.get("DeltaF_local_min", np.inf)),
                int(row.get("minimum_rank", 0)),
            )

        global_best = sorted(
            [row for row in mandatory if "global_best" in str(row.get("mandatory_basin_reasons", "")).split(";")],
            key=lambda row: float(row.get("DeltaF_local_min", np.inf)),
        )
        edge_risk = sorted(
            [row for row in mandatory if "edge_risk" in str(row.get("mandatory_basin_reasons", "")).split(";")],
            key=edge_key,
        )
        delta_near_eps = sorted(
            [row for row in mandatory if "Delta_near_epsilon" in str(row.get("mandatory_basin_reasons", "")).split(";")],
            key=delta_key,
        )
        near_degenerate = sorted(
            [row for row in mandatory if "near_degenerate" in str(row.get("mandatory_basin_reasons", "")).split(";")],
            key=energy_key,
        )
        other_mandatory = sorted(
            [
                row
                for row in mandatory
                if not (
                    {"global_best", "edge_risk", "Delta_near_epsilon", "near_degenerate"}
                    & set(str(row.get("mandatory_basin_reasons", "")).split(";"))
                )
            ],
            key=energy_key,
        )

        append_ranked(global_best, 1, "global_best")
        append_ranked(edge_risk, max_edge_risk_basins, "edge_risk_rank_and_cap")
        append_ranked(delta_near_eps, max_delta_near_eps_basins, "delta_near_eps_rank_and_cap")
        append_ranked(near_degenerate, max_near_degenerate_basins, "near_degenerate_rank_and_cap")
        append_ranked(other_mandatory, total_cap, "other_mandatory_rank_and_cap")

        selected_mandatory_count = int(sum(1 for row in selected if bool(row.get("mandatory_basin", False))))
        mandatory_overflow_count = max(0, int(len(mandatory)) - selected_mandatory_count)
        mandatory_overflow = bool(mandatory_overflow_count > 0)
        for row in selected:
            row["mandatory_overflow"] = mandatory_overflow
            row["mandatory_overflow_count"] = int(mandatory_overflow_count)

        optional_slots = max(0, min(int(max_optional_refined_basins), total_cap - len(selected)))
        for row in ordinary[:optional_slots]:
            row["mandatory_overflow_policy"] = "rank_and_cap"
            row["mandatory_overflow"] = mandatory_overflow
            row["mandatory_overflow_count"] = int(mandatory_overflow_count)
            row["rank_and_cap_selection_reason"] = "ordinary_optional"
            selected.append(row)
        return sorted(selected, key=lambda r: float(r.get("DeltaF_local_min", np.inf)))

    selected: list[dict[str, object]] = list(mandatory)
    if mandatory_basins_can_exceed_cap:
        optional_slots = max(0, int(max_optional_refined_basins))
    else:
        optional_slots = max(0, min(int(max_optional_refined_basins), int(max_total_refined_basins) - len(selected)))
        if len(selected) > int(max_total_refined_basins):
            selected = selected[: int(max_total_refined_basins)]
            optional_slots = 0
    selected.extend(ordinary[:optional_slots])
    return sorted(selected, key=lambda r: float(r.get("DeltaF_local_min", np.inf)))


def mark_energy_window_pruning(
    rows: list[dict[str, object]],
    local_refine_energy_window: float,
    delta_eps: float,
    delta_refine_half_width: float,
    enabled: bool = False,
) -> list[dict[str, object]]:
    """Mark ordinary high-energy basins as pruned; mandatory basins are kept."""
    marked: list[dict[str, object]] = []
    for row_in in rows:
        row = dict(row_in)
        reasons = _mandatory_basin_reasons(row, delta_eps, delta_refine_half_width)
        row["mandatory_basin"] = bool(reasons)
        row["mandatory_basin_reasons"] = ";".join(sorted(reasons)) if reasons else str(row.get("mandatory_basin_reasons", "none"))
        row["basin_selection_reason"] = _local_refine_selection_reason(row, delta_eps, delta_refine_half_width)
        energy_above = float(row.get("energy_above_global", np.nan))
        should_prune = bool(
            enabled
            and not reasons
            and np.isfinite(energy_above)
            and energy_above > float(local_refine_energy_window)
        )
        row["pruned_by_energy_window"] = bool(should_prune)
        if should_prune:
            row["pruned_reason"] = "ordinary_above_energy_window"
        else:
            row["pruned_reason"] = str(row.get("pruned_reason", "not_pruned"))
        marked.append(row)
    return marked


def branch_reuse_signature(values: dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_branch_reuse_candidate(
    cached_branch: dict[str, object] | None,
    candidate: dict[str, object],
    solver_config_signature: str,
    local_box_signature: str,
    q_tolerance: float,
    delta_tolerance: float,
    energy_tolerance: float,
    lower_competing_deltaf: float | None = None,
) -> dict[str, object]:
    """Decide whether a cached refined branch may be reused without silence."""
    if cached_branch is None:
        return {"reuse_allowed": False, "reuse_rejection_reason": "missing_cached_branch"}
    if str(cached_branch.get("solver_config_signature", "")) != str(solver_config_signature):
        return {"reuse_allowed": False, "reuse_rejection_reason": "solver_config_signature_mismatch"}
    if str(cached_branch.get("local_box_signature", "")) != str(local_box_signature):
        return {"reuse_allowed": False, "reuse_rejection_reason": "local_box_signature_mismatch"}

    cached_q = float(cached_branch.get("q_refined", np.nan))
    cached_delta = float(cached_branch.get("Delta_refined", np.nan))
    cached_deltaf = float(cached_branch.get("DeltaF_refined", np.nan))
    candidate_q = float(candidate.get("q_local_min", np.nan))
    candidate_delta = float(candidate.get("Delta_local_min", np.nan))
    candidate_deltaf = float(candidate.get("DeltaF_local_min", np.nan))
    if not np.isfinite([cached_q, cached_delta, cached_deltaf, candidate_q, candidate_delta, candidate_deltaf]).all():
        return {"reuse_allowed": False, "reuse_rejection_reason": "nonfinite_reuse_inputs"}
    if abs(cached_q - candidate_q) > float(q_tolerance):
        return {"reuse_allowed": False, "reuse_rejection_reason": "q_mismatch"}
    if abs(cached_delta - candidate_delta) > float(delta_tolerance):
        return {"reuse_allowed": False, "reuse_rejection_reason": "Delta_mismatch"}
    if abs(cached_deltaf - candidate_deltaf) > float(energy_tolerance):
        return {"reuse_allowed": False, "reuse_rejection_reason": "energy_mismatch"}
    if lower_competing_deltaf is not None and np.isfinite(float(lower_competing_deltaf)):
        if float(lower_competing_deltaf) < cached_deltaf - float(energy_tolerance):
            return {"reuse_allowed": False, "reuse_rejection_reason": "lower_energy_competing_branch"}
    return {
        "reuse_allowed": True,
        "reuse_rejection_reason": "reuse_allowed",
        "refined_q": float(cached_q),
        "refined_Delta": float(cached_delta),
        "refined_DeltaF": float(cached_deltaf),
    }


def build_branch_reuse_cache_record(
    refined_branch: dict[str, object],
    candidate: dict[str, object],
    solver_config_signature: str,
    local_box_signature: str,
    *,
    point_id: int = -1,
    q_window_level: int = 0,
) -> dict[str, object]:
    """Build the explicit cache row that a future reuse integration may persist."""
    q_refined = float(refined_branch.get("refined_q", refined_branch.get("q_refined", candidate.get("q_local_min", np.nan))))
    delta_refined = float(
        refined_branch.get("refined_Delta", refined_branch.get("Delta_refined", candidate.get("Delta_local_min", np.nan)))
    )
    deltaf_refined = float(
        refined_branch.get("refined_DeltaF", refined_branch.get("DeltaF_refined", candidate.get("DeltaF_local_min", np.nan)))
    )
    branch_id = int(candidate.get("minimum_rank", candidate.get("branch_id", -1)))
    basin_id = int(candidate.get("basin_id", branch_id))
    signatures_present = bool(str(solver_config_signature)) and bool(str(local_box_signature))
    cache_valid = bool(np.isfinite([q_refined, delta_refined, deltaf_refined]).all() and signatures_present)
    return {
        "point_id": int(point_id),
        "branch_id": int(branch_id),
        "basin_id": int(basin_id),
        "cluster_id": int(basin_id),
        "q_refined": float(q_refined),
        "Delta_refined": float(delta_refined),
        "DeltaF_refined": float(deltaf_refined),
        "q_window_level": int(q_window_level),
        "solver_config_signature": str(solver_config_signature),
        "local_box_signature": str(local_box_signature),
        "reuse_cache_valid": int(cache_valid),
    }


def build_branch_reuse_diagnostic_record(
    cached_branch: dict[str, object] | None,
    candidate: dict[str, object],
    decision: dict[str, object],
    solver_config_signature: str,
    local_box_signature: str,
    *,
    lower_competing_deltaf: float | None = None,
) -> dict[str, object]:
    """Build an auditable reuse decision row without performing reuse."""
    candidate_q = float(candidate.get("q_local_min", np.nan))
    candidate_delta = float(candidate.get("Delta_local_min", np.nan))
    candidate_deltaf = float(candidate.get("DeltaF_local_min", np.nan))
    cached_q = float(cached_branch.get("q_refined", np.nan)) if cached_branch is not None else float("nan")
    cached_delta = float(cached_branch.get("Delta_refined", np.nan)) if cached_branch is not None else float("nan")
    cached_deltaf = float(cached_branch.get("DeltaF_refined", np.nan)) if cached_branch is not None else float("nan")
    branch_id = int(candidate.get("minimum_rank", candidate.get("branch_id", -1)))
    basin_id = int(candidate.get("basin_id", branch_id))

    def finite_abs_diff(left: float, right: float) -> float:
        if np.isfinite([left, right]).all():
            return float(abs(left - right))
        return float("nan")

    return {
        "branch_id": int(branch_id),
        "basin_id": int(basin_id),
        "reuse_attempted": int(cached_branch is not None),
        "reuse_allowed": int(bool(decision.get("reuse_allowed", False))),
        "reuse_rejection_reason": str(decision.get("reuse_rejection_reason", "missing_reuse_decision")),
        "solver_config_signature": str(solver_config_signature),
        "local_box_signature": str(local_box_signature),
        "cached_solver_config_signature": str(cached_branch.get("solver_config_signature", "")) if cached_branch is not None else "",
        "cached_local_box_signature": str(cached_branch.get("local_box_signature", "")) if cached_branch is not None else "",
        "candidate_q": float(candidate_q),
        "candidate_Delta": float(candidate_delta),
        "candidate_DeltaF": float(candidate_deltaf),
        "cached_q": float(cached_q),
        "cached_Delta": float(cached_delta),
        "cached_DeltaF": float(cached_deltaf),
        "q_abs_diff": finite_abs_diff(cached_q, candidate_q),
        "Delta_abs_diff": finite_abs_diff(cached_delta, candidate_delta),
        "energy_abs_diff": finite_abs_diff(cached_deltaf, candidate_deltaf),
        "lower_competing_deltaf": float(lower_competing_deltaf) if lower_competing_deltaf is not None else float("nan"),
    }


def _delta_boundary_ambiguous(
    result: PointOracleResult,
    delta_eps: float,
    delta_boundary_margin: float,
    free_energy_ambiguity_tol: float,
    positive_delta_gap: float | None = None,
) -> bool:
    if result.delta_opt < float(delta_eps):
        if positive_delta_gap is None or not np.isfinite(positive_delta_gap):
            return False
        return bool(abs(float(positive_delta_gap)) <= float(free_energy_ambiguity_tol))
    return bool(
        abs(result.delta_opt - float(delta_eps)) <= float(delta_boundary_margin)
        or abs(result.omega_global) <= float(free_energy_ambiguity_tol)
    )


def _finite_result(result: PointOracleResult) -> bool:
    return bool(
        np.isfinite(
            [
                result.eta,
                result.ic_plus,
                result.ic_minus,
                result.q_opt,
                result.delta_opt,
                result.omega_global,
            ]
        ).all()
    )


def _phase_candidate(result: PointOracleResult, delta_eps: float, delta_ambiguous: bool) -> int:
    if delta_ambiguous:
        return PHASE_AMBIGUOUS
    if result.delta_opt < float(delta_eps):
        return PHASE_NORMAL
    return PHASE_SUPERCONDUCTING


def _q_hit_side(result: PointOracleResult) -> str:
    if abs(result.q_opt - result.q_min) <= abs(result.q_max - result.q_opt):
        return "min"
    return "max"


def expand_q_config(
    cfg: EtaPhaseConfig,
    result: PointOracleResult,
    expand_factor: float,
    pad_steps: int,
    q_max_abs: float,
) -> EtaPhaseConfig:
    width = float(cfg.q_max - cfg.q_min)
    if width <= 0:
        raise ValueError("q_max must be greater than q_min")
    old_step = width / max(1, int(cfg.n_q) - 1)
    half_extra = max(0.5 * (float(expand_factor) - 1.0) * width, abs(old_step) * int(pad_steps))

    new_min = float(cfg.q_min)
    new_max = float(cfg.q_max)
    hit_side = _q_hit_side(result)
    if hit_side == "min":
        new_min = max(-abs(float(q_max_abs)), new_min - half_extra)
    else:
        new_max = min(abs(float(q_max_abs)), new_max + half_extra)

    new_width = max(new_max - new_min, old_step)
    new_n_q = max(int(cfg.n_q), int(math.ceil(new_width / max(abs(old_step), 1e-12))) + 1)
    return replace(cfg, q_min=new_min, q_max=new_max, n_q=new_n_q)


def refine_delta_config(
    cfg: EtaPhaseConfig,
    result: PointOracleResult,
    delta_eps: float,
    half_width: float,
    n_delta_refined: int,
    refinement_level: int,
) -> EtaPhaseConfig:
    center = max(float(result.delta_opt), float(delta_eps))
    width = float(half_width) / max(1, int(refinement_level))
    lo = max(0.0, center - width)
    hi = min(float(cfg.delta_max), center + width)
    if hi <= lo:
        hi = min(float(cfg.delta_max), lo + max(float(delta_eps), 1e-6))
    if lo <= float(delta_eps) <= hi:
        lo = 0.0
    return replace(cfg, delta_min=lo, delta_max=hi, n_delta=max(3, int(n_delta_refined)))


def positive_delta_config(cfg: EtaPhaseConfig, delta_eps: float, n_delta_positive: int) -> EtaPhaseConfig:
    lo = max(float(delta_eps), float(cfg.delta_min))
    hi = float(cfg.delta_max)
    if hi <= lo:
        hi = lo + max(float(delta_eps), 1e-6)
    return replace(cfg, delta_min=lo, delta_max=hi, n_delta=max(3, int(n_delta_positive)))


def _status_name(status_code: int) -> str:
    if int(status_code) == 0:
        return "trusted"
    names: list[str] = []
    if status_code & STATUS_Q_EDGE_UNRESOLVED:
        names.append("q_edge_unresolved")
    if status_code & STATUS_DELTA_BOUNDARY_UNRESOLVED:
        names.append("delta_boundary_unresolved")
    if status_code & STATUS_NONFINITE_OUTPUT:
        names.append("nonfinite_output")
    if status_code & STATUS_MAX_Q_REFINEMENT_REACHED:
        names.append("max_q_refinement_reached")
    if status_code & STATUS_MAX_DELTA_REFINEMENT_REACHED:
        names.append("max_delta_refinement_reached")
    return ";".join(names) if names else "unknown_nonzero_status"


def _confirm_one_point_legacy(
    kT: float,
    JA: float,
    base_cfg: EtaPhaseConfig,
    device: torch.device,
    delta_eps: float,
    delta_boundary_margin: float,
    q_edge_margin: float | None,
    free_energy_ambiguity_tol: float,
    positive_delta_gap_tol: float,
    enable_q_expansion: bool,
    q_expand_factor: float,
    q_expand_pad_steps: int,
    q_max_abs: float,
    max_q_refinements: int,
    enable_delta_refinement: bool,
    delta_refine_half_width: float,
    n_delta_refined: int,
    max_delta_refinements: int,
    allow_ambiguous_output: bool,
) -> ConfirmedPoint:
    cfg_current = base_cfg
    result = evaluate_one_point_once(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
    q_ref_level = 0
    delta_ref_level = 0
    q_expanded = 0
    delta_refined = 0
    n_delta_refined_used = 0
    q_status = Q_ACTIVE
    delta_status = DELTA_STABLE
    q_edge_hit = 0
    q_unresolved = 0
    delta_unresolved = 0
    status_code = 0
    positive_delta_gap = float("nan")
    positive_delta_checked = 0

    finite_outputs = _finite_result(result)

    def resolve_normal_delta_candidate(
        current_result: PointOracleResult,
        current_cfg: EtaPhaseConfig,
    ) -> tuple[PointOracleResult, bool, bool]:
        nonlocal positive_delta_gap, positive_delta_checked
        nonlocal delta_refined, delta_ref_level, n_delta_refined_used

        current_finite = _finite_result(current_result)
        if not current_finite:
            return current_result, False, True

        if current_result.delta_opt < float(delta_eps) and enable_delta_refinement:
            positive_delta_checked = 1
            if delta_ref_level == 0:
                delta_ref_level = 1
            delta_refined = 1
            n_delta_refined_used = int(n_delta_refined)
            pos_cfg = positive_delta_config(current_cfg, delta_eps=delta_eps, n_delta_positive=n_delta_refined)
            pos_result = evaluate_one_point_once(kT, JA, pos_cfg, device, q_edge_margin=q_edge_margin)
            pos_finite = _finite_result(pos_result)
            positive_delta_gap = float(pos_result.omega_global) if pos_finite else float("nan")
            if pos_finite and positive_delta_gap < -float(positive_delta_gap_tol):
                current_result = pos_result
                current_finite = True
            elif pos_finite and positive_delta_gap > float(positive_delta_gap_tol):
                delta_ambiguous = False
                return current_result, current_finite, delta_ambiguous
            else:
                delta_ambiguous = True
                return current_result, current_finite, delta_ambiguous

        delta_ambiguous = _delta_boundary_ambiguous(
            current_result,
            delta_eps=delta_eps,
            delta_boundary_margin=delta_boundary_margin,
            free_energy_ambiguity_tol=free_energy_ambiguity_tol,
            positive_delta_gap=None,
        )
        return current_result, current_finite, delta_ambiguous

    result, finite_outputs, delta_amb = resolve_normal_delta_candidate(result, cfg_current)
    phase_candidate = _phase_candidate(result, delta_eps=delta_eps, delta_ambiguous=delta_amb)
    if not finite_outputs:
        status_code |= STATUS_NONFINITE_OUTPUT

    if result.delta_opt < float(delta_eps):
        q_status = Q_NOT_APPLICABLE
        q_edge_hit = 0
    else:
        q_edge_hit = int(result.q_edge_hit_raw)
        q_status = Q_EDGE_HIT if q_edge_hit else Q_ACTIVE

    while (
        finite_outputs
        and result.delta_opt >= float(delta_eps)
        and result.q_edge_hit_raw
        and enable_q_expansion
        and q_ref_level < int(max_q_refinements)
    ):
        next_cfg = expand_q_config(
            cfg_current,
            result,
            expand_factor=q_expand_factor,
            pad_steps=q_expand_pad_steps,
            q_max_abs=q_max_abs,
        )
        if next_cfg.q_min == cfg_current.q_min and next_cfg.q_max == cfg_current.q_max:
            break
        cfg_current = next_cfg
        q_ref_level += 1
        q_expanded = 1
        result = evaluate_one_point_once(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
        result, finite_outputs, delta_amb = resolve_normal_delta_candidate(result, cfg_current)

    if result.delta_opt < float(delta_eps):
        q_status = Q_NOT_APPLICABLE
        q_edge_hit = 0
    else:
        q_edge_hit = int(result.q_edge_hit_raw)
        if q_edge_hit:
            q_status = Q_UNRESOLVED
            q_unresolved = 1
            status_code |= STATUS_Q_EDGE_UNRESOLVED
            if q_ref_level >= int(max_q_refinements):
                status_code |= STATUS_MAX_Q_REFINEMENT_REACHED
        elif q_expanded:
            q_status = Q_EXPANDED_CONFIRMED
        else:
            q_status = Q_ACTIVE

    while (
        finite_outputs
        and delta_amb
        and enable_delta_refinement
        and delta_ref_level < int(max_delta_refinements)
    ):
        delta_ref_level += 1
        delta_refined = 1
        n_delta_refined_used = int(n_delta_refined)
        delta_cfg = refine_delta_config(
            cfg_current,
            result,
            delta_eps=delta_eps,
            half_width=delta_refine_half_width,
            n_delta_refined=n_delta_refined,
            refinement_level=delta_ref_level,
        )
        result = evaluate_one_point_once(kT, JA, delta_cfg, device, q_edge_margin=q_edge_margin)
        cfg_current = delta_cfg
        result, finite_outputs, delta_amb = resolve_normal_delta_candidate(result, cfg_current)

        if finite_outputs and result.delta_opt >= float(delta_eps) and result.q_edge_hit_raw and enable_q_expansion:
            while result.q_edge_hit_raw and q_ref_level < int(max_q_refinements):
                next_cfg = expand_q_config(
                    cfg_current,
                    result,
                    expand_factor=q_expand_factor,
                    pad_steps=q_expand_pad_steps,
                    q_max_abs=q_max_abs,
                )
                if next_cfg.q_min == cfg_current.q_min and next_cfg.q_max == cfg_current.q_max:
                    break
                cfg_current = next_cfg
                q_ref_level += 1
                q_expanded = 1
                result = evaluate_one_point_once(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
                result, finite_outputs, delta_amb = resolve_normal_delta_candidate(result, cfg_current)
                if not finite_outputs:
                    break

    if not finite_outputs:
        status_code |= STATUS_NONFINITE_OUTPUT

    if result.delta_opt < float(delta_eps):
        q_status = Q_NOT_APPLICABLE
        q_edge_hit = 0
        q_unresolved = 0
        status_code &= ~STATUS_Q_EDGE_UNRESOLVED
        status_code &= ~STATUS_MAX_Q_REFINEMENT_REACHED
    else:
        q_edge_hit = int(result.q_edge_hit_raw)
        if q_edge_hit:
            q_status = Q_UNRESOLVED
            q_unresolved = 1
            status_code |= STATUS_Q_EDGE_UNRESOLVED
            if q_ref_level >= int(max_q_refinements):
                status_code |= STATUS_MAX_Q_REFINEMENT_REACHED
        elif q_expanded:
            q_status = Q_EXPANDED_CONFIRMED
        else:
            q_status = Q_ACTIVE

    delta_amb_int = int(delta_amb)
    if delta_amb:
        if delta_refined and not allow_ambiguous_output:
            delta_status = DELTA_UNRESOLVED
            delta_unresolved = 1
            status_code |= STATUS_DELTA_BOUNDARY_UNRESOLVED
            if delta_ref_level >= int(max_delta_refinements):
                status_code |= STATUS_MAX_DELTA_REFINEMENT_REACHED
        else:
            delta_status = DELTA_BOUNDARY_AMBIGUOUS
            if not allow_ambiguous_output:
                delta_unresolved = 1
                status_code |= STATUS_DELTA_BOUNDARY_UNRESOLVED
    elif delta_refined:
        delta_status = DELTA_REFINED_CONFIRMED
    else:
        delta_status = DELTA_STABLE

    phase_candidate = _phase_candidate(result, delta_eps=delta_eps, delta_ambiguous=bool(delta_amb_int))
    _ = allow_ambiguous_output
    trusted = int(status_code == 0 and finite_outputs and not q_unresolved and not delta_unresolved and not delta_amb_int)

    return ConfirmedPoint(
        kT=float(kT),
        JA=float(JA),
        eta=float(result.eta),
        q_opt=float(result.q_opt),
        delta_opt=float(result.delta_opt),
        ic_plus=float(result.ic_plus),
        ic_minus=float(result.ic_minus),
        phase_candidate=int(phase_candidate),
        q_status=int(q_status),
        q_min=float(result.q_min),
        q_max=float(result.q_max),
        n_q=int(result.n_q),
        q_index=int(result.q_index),
        q_edge_distance=float(result.q_edge_distance),
        q_edge_hit=int(q_edge_hit),
        q_refinement_level=int(q_ref_level),
        q_expanded=int(q_expanded),
        q_unresolved=int(q_unresolved),
        delta_status=int(delta_status),
        delta_min=float(result.delta_min),
        delta_max=float(result.delta_max),
        n_delta=int(result.n_delta),
        n_delta_refined=int(n_delta_refined_used),
        delta_refinement_level=int(delta_ref_level),
        delta_boundary_ambiguous=int(delta_amb_int),
        delta_refined=int(delta_refined),
        delta_unresolved=int(delta_unresolved),
        free_energy_gap_to_normal=float(result.omega_global),
        positive_delta_gap=float(positive_delta_gap),
        positive_delta_checked=int(positive_delta_checked),
        exact_status_code=int(status_code),
        exact_status_name=_status_name(int(status_code)),
        trusted_exact=int(trusted),
        oracle_mode="legacy",
        search_mode="legacy",
        initial_q_min=float(base_cfg.q_min),
        initial_q_max=float(base_cfg.q_max),
        final_q_min=float(result.q_min),
        final_q_max=float(result.q_max),
        initial_n_q=int(base_cfg.n_q),
        final_n_q=int(result.n_q),
        initial_dq=float((float(base_cfg.q_max) - float(base_cfg.q_min)) / max(1, int(base_cfg.n_q) - 1)),
        final_dq=float((float(result.q_max) - float(result.q_min)) / max(1, int(result.n_q) - 1)),
        q_expansion_count=int(q_ref_level),
        q_expansion_directions=("legacy_q_expand" if q_expanded else "none"),
        q_expansion_trigger=("q_edge_hit" if q_expanded else "none"),
        q_window_coverage_valid=int(not q_unresolved),
        q_window_unresolved=int(q_unresolved),
        qopt_edge_hit_initial=int(result.q_edge_hit_raw if not q_expanded else 0),
        qopt_edge_hit_final=int(q_edge_hit),
        edge_risk_left_initial=0,
        edge_risk_right_initial=0,
        edge_risk_left_final=0,
        edge_risk_right_final=0,
        expanded_window_found_lower_branch=0,
        phase_changed_after_q_expansion=0,
        local_minima_count=0,
        refined_local_minima_count=0,
        near_degenerate_branch_count=0,
        selected_minimum_rank=1,
        branch_candidates_path="N/A",
        delta_refinement_triggered=int(delta_refined),
        delta_refinement_valid=int(not delta_unresolved),
        boundary_ambiguous=int(delta_amb_int),
        changed_after_delta_refinement=0,
        unresolved_reason=("" if status_code == 0 else _status_name(int(status_code))),
    )


def _write_branch_candidates_csv(path: Path, point_id: int, kT: float, JA: float, rows: list[dict[str, object]]) -> None:
    if not rows:
        pd.DataFrame(
            columns=[
                "point_id",
                "kBT",
                "JA",
                "branch_id",
                "q_local_min",
                "Delta_local_min",
                "DeltaF_local_min",
                "energy_above_global",
                "refined_q",
                "refined_Delta",
                "refined_DeltaF",
                "refined_status",
                "distance_to_q_edge",
                "basin_id",
                "cluster_size",
                "merged_branch_ids",
                "cluster_reason",
                "basin_risk_flags",
                "basin_has_global_best",
                "basin_has_edge_risk",
                "basin_has_delta_near_epsilon",
                "basin_has_near_degenerate",
                "basin_has_low_energy_window",
                "basin_is_ordinary",
                "basin_global_best_candidate_count",
                "basin_edge_risk_candidate_count",
                "basin_delta_near_epsilon_candidate_count",
                "basin_near_degenerate_candidate_count",
                "basin_low_energy_window_candidate_count",
                "mandatory_basin_reasons",
                "basin_selection_reason",
                "selected_for_refinement",
                "dropped_mandatory_reason",
                "mandatory_overflow_policy",
                "mandatory_overflow",
                "mandatory_overflow_count",
                "rank_and_cap_selection_reason",
                "pruned_reason",
                "basin_q_width",
                "basin_Delta_width",
                "basin_energy_span",
                "basin_curvature_proxy",
                "topology_pending",
            ]
        ).to_csv(path, index=False)
        return
    out_rows: list[dict[str, object]] = []
    for i, row in enumerate(rows, start=1):
        out_rows.append(
            {
                "point_id": int(point_id),
                "kBT": float(kT),
                "JA": float(JA),
                "branch_id": int(i),
                "q_local_min": float(row.get("q_local_min", np.nan)),
                "Delta_local_min": float(row.get("Delta_local_min", np.nan)),
                "DeltaF_local_min": float(row.get("DeltaF_local_min", np.nan)),
                "energy_above_global": float(row.get("energy_above_global", np.nan)),
                "refined_q": float(row.get("refined_q", row.get("q_local_min", np.nan))),
                "refined_Delta": float(row.get("refined_Delta", row.get("Delta_local_min", np.nan))),
                "refined_DeltaF": float(row.get("refined_DeltaF", row.get("DeltaF_local_min", np.nan))),
                "refined_status": str(row.get("refined_status", "N/A")),
                "distance_to_q_edge": float(row.get("distance_to_q_edge", np.nan)),
                "basin_id": int(row.get("basin_id", i)),
                "cluster_size": int(row.get("cluster_size", 1)),
                "merged_branch_ids": str(row.get("merged_branch_ids", row.get("minimum_rank", i))),
                "cluster_reason": str(row.get("cluster_reason", "not_clustered")),
                "basin_risk_flags": str(row.get("basin_risk_flags", "not_annotated")),
                "basin_has_global_best": int(bool(row.get("basin_has_global_best", int(row.get("minimum_rank", i)) == 1))),
                "basin_has_edge_risk": int(bool(row.get("basin_has_edge_risk", row.get("edge_risk", False)))),
                "basin_has_delta_near_epsilon": int(bool(row.get("basin_has_delta_near_epsilon", False))),
                "basin_has_near_degenerate": int(bool(row.get("basin_has_near_degenerate", False))),
                "basin_has_low_energy_window": int(bool(row.get("basin_has_low_energy_window", row.get("within_low_energy_window", False)))),
                "basin_is_ordinary": int(bool(row.get("basin_is_ordinary", False))),
                "basin_global_best_candidate_count": int(row.get("basin_global_best_candidate_count", int(int(row.get("minimum_rank", i)) == 1))),
                "basin_edge_risk_candidate_count": int(row.get("basin_edge_risk_candidate_count", int(bool(row.get("edge_risk", False))))),
                "basin_delta_near_epsilon_candidate_count": int(row.get("basin_delta_near_epsilon_candidate_count", 0)),
                "basin_near_degenerate_candidate_count": int(row.get("basin_near_degenerate_candidate_count", 0)),
                "basin_low_energy_window_candidate_count": int(row.get("basin_low_energy_window_candidate_count", int(bool(row.get("within_low_energy_window", False))))),
                "mandatory_basin_reasons": str(row.get("mandatory_basin_reasons", "none")),
                "basin_selection_reason": str(
                    row.get(
                        "basin_selection_reason",
                        row.get("mandatory_basin_reasons", "ordinary_selected"),
                    )
                ),
                "selected_for_refinement": int(row.get("selected_for_refinement", int(str(row.get("refined_status", "")) == "refined_box"))),
                "dropped_mandatory_reason": str(row.get("dropped_mandatory_reason", "not_recorded")),
                "mandatory_overflow_policy": str(row.get("mandatory_overflow_policy", "keep_all")),
                "mandatory_overflow": int(bool(row.get("mandatory_overflow", False))),
                "mandatory_overflow_count": int(row.get("mandatory_overflow_count", 0)),
                "rank_and_cap_selection_reason": str(row.get("rank_and_cap_selection_reason", "not_recorded")),
                "pruned_reason": str(row.get("pruned_reason", "not_pruned")),
                "basin_q_width": float(row.get("basin_q_width", 0.0)),
                "basin_Delta_width": float(row.get("basin_Delta_width", 0.0)),
                "basin_energy_span": float(row.get("basin_energy_span", 0.0)),
                "basin_curvature_proxy": float(row.get("basin_curvature_proxy", 0.0)),
                "topology_pending": int(1),
            }
        )
    pd.DataFrame(out_rows).to_csv(path, index=False)


LOCAL_BOX_RECORD_FIELDS = [
    "point_id",
    "kT",
    "JA",
    "branch_id",
    "branch_rank_before_refine",
    "grid_index",
    "candidate_type",
    "selection_reason",
    "basin_id",
    "cluster_size",
    "merged_branch_ids",
    "basin_risk_flags",
    "basin_selection_reason",
    "mandatory_basin_reasons",
    "basin_has_global_best",
    "basin_has_edge_risk",
    "basin_has_delta_near_epsilon",
    "basin_has_near_degenerate",
    "basin_has_low_energy_window",
    "basin_is_ordinary",
    "q_center_before_refine",
    "Delta_center_before_refine",
    "DeltaF_before_refine",
    "energy_above_global_before_refine",
    "distance_to_q_edge_before_refine",
    "box_q_min",
    "box_q_max",
    "box_Delta_min",
    "box_Delta_max",
    "n_q_local",
    "n_Delta_local",
    "local_grid_evaluations",
    "box_runtime_sec",
    "refined_q",
    "refined_Delta",
    "refined_DeltaF",
    "refined_status",
    "changed_global_minimum",
    "changed_phase_label",
    "near_degenerate_after_refine",
    "q_window_level",
    "incremental_qexpansion_used",
    "reused_from_previous_scan",
    "pruned_reason",
]


def _phase_from_deltaf(delta: float, deltaf: float, delta_eps: float, free_energy_ambiguity_tol: float) -> int:
    if np.isfinite(deltaf) and np.isfinite(delta) and deltaf < -float(free_energy_ambiguity_tol) and delta > float(delta_eps):
        return PHASE_SUPERCONDUCTING
    if np.isfinite(delta) and delta < float(delta_eps):
        return PHASE_NORMAL
    if np.isfinite(deltaf) and abs(deltaf) <= float(free_energy_ambiguity_tol):
        return PHASE_AMBIGUOUS
    return PHASE_NORMAL


def _local_refine_selection_reason(row: dict[str, object], delta_eps: float, delta_refine_half_width: float) -> str:
    mandatory_reasons = _mandatory_basin_reasons(row, delta_eps, delta_refine_half_width)
    reasons: list[str] = list(mandatory_reasons)
    if bool(row.get("within_low_energy_window", False)) or _truthy_flag(row, "basin_has_low_energy_window"):
        reasons.append("within_low_energy_window")
    if "near_degenerate" in mandatory_reasons and "within_low_energy_window" not in reasons:
        reasons.append("within_low_energy_window")
    if not reasons and int(row["minimum_rank"]) == 1:
        reasons.append("global_best")
    return ";".join(sorted(set(reasons))) if reasons else "ordinary_selected"


def _write_local_box_records_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=LOCAL_BOX_RECORD_FIELDS)
    table.to_csv(path, index=False)


def _write_local_refinement_summary_json(path: Path, rows: list[dict[str, object]], n_points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = [float(r.get("box_runtime_sec", 0.0)) for r in rows]
    grid_evals = [int(r.get("local_grid_evaluations", 0)) for r in rows]
    summary = {
        "n_points": int(n_points),
        "local_box_rows": int(len(rows)),
        "local_box_runtime_sec_sum": float(np.sum(runtime)) if runtime else 0.0,
        "local_box_runtime_sec_mean": float(np.mean(runtime)) if runtime else 0.0,
        "local_grid_evaluations_sum": int(np.sum(grid_evals)) if grid_evals else 0,
        "changed_global_minimum_count": int(sum(int(r.get("changed_global_minimum", 0)) for r in rows)),
        "changed_phase_label_count": int(sum(int(r.get("changed_phase_label", 0)) for r in rows)),
        "reused_box_count": int(sum(int(r.get("reused_from_previous_scan", 0)) for r in rows)),
        "pruned_box_count": int(sum(1 for r in rows if str(r.get("pruned_reason", "")) not in {"", "not_pruned_stage_1"})),
        "instrumentation_stage": "stage_1_box_level_logging_only",
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _confirm_one_point_robust(
    kT: float,
    JA: float,
    base_cfg: EtaPhaseConfig,
    device: torch.device,
    delta_eps: float,
    q_eps: float,
    q_edge_margin: float | None,
    free_energy_ambiguity_tol: float,
    positive_delta_gap_tol: float,
    q_max_abs: float,
    max_q_expansion_levels: int,
    local_refine_energy_window: float,
    max_refined_minima: int,
    local_refine_q_half_width: float,
    local_refine_delta_half_width: float,
    n_q_refined: int,
    n_delta_refined: int,
    enable_delta_refinement: bool,
    delta_refine_half_width: float,
    max_delta_refinements: int,
    branch_dir: Path | None,
    point_index: int,
    use_incremental_q_expansion: bool = False,
    local_box_records: list[dict[str, object]] | None = None,
    enable_basin_clustering: bool = False,
    basin_q_cluster_factor: float = 1.5,
    basin_delta_cluster_factor: float = 1.5,
    basin_energy_cluster_factor: float = 1.0,
    enable_selective_refinement: bool = False,
    max_optional_refined_basins: int = 3,
    mandatory_basins_can_exceed_cap: bool = True,
    high_risk_overflow_policy: str = "keep_all",
    max_edge_risk_basins: int = 1,
    max_delta_near_eps_basins: int = 2,
    max_near_degenerate_basins: int = 2,
    energy_window_pruning_enabled: bool = False,
    local_refine_pruning_energy_window: float | None = None,
) -> ConfirmedPoint:
    point_t0 = time.perf_counter()
    base_scan_runtime_sec = 0.0
    q_expansion_runtime_sec = 0.0
    q_expansion_left_runtime_sec = 0.0
    q_expansion_right_runtime_sec = 0.0
    delta_refinement_runtime_sec = 0.0
    local_refinement_runtime_sec = 0.0
    merge_cache_runtime_sec = 0.0
    local_minima_detection_runtime_sec = 0.0
    fallback_full_rescan_runtime_sec = 0.0
    base_q_points_evaluated = 0
    added_left_q_points_evaluated = 0
    added_right_q_points_evaluated = 0
    recomputed_q_points = 0
    base_grid_evaluations = 0
    incremental_q_grid_evaluations = 0
    fallback_full_rescan_grid_evaluations = 0
    delta_refinement_grid_evaluations = 0
    local_refinement_grid_evaluations = 0
    incremental_expansion_used = 0
    fallback_full_rescan_used = 0
    fallback_full_rescan_reason = "N/A"

    cfg_current = replace(base_cfg)
    initial_scan = _run_scan_with_normal(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
    base_scan_runtime_sec += float(initial_scan.scan_runtime_sec)
    base_q_points_evaluated += int(initial_scan.q_points_evaluated)
    base_grid_evaluations += int(initial_scan.estimated_grid_evaluations)
    initial_diag = _diagnose_q_window(
        initial_scan,
        delta_eps=delta_eps,
        ambiguity_tol=free_energy_ambiguity_tol,
    )
    scans: list[ScanResult] = [initial_scan]
    expansion_directions: list[str] = []
    expansion_triggers: list[str] = []
    sym_used = False

    for _lvl in range(int(max_q_expansion_levels)):
        current = scans[-1]
        diag = _diagnose_q_window(current, delta_eps=delta_eps, ambiguity_tol=free_energy_ambiguity_tol)
        direction, trigger = _select_expansion_direction(diag, allow_symmetric_once=(not sym_used))
        if direction == "both":
            sym_used = True
        if direction == "none":
            break
        next_cfg = _expand_cfg_keep_density(cfg_current, direction=direction, q_max_abs=q_max_abs)
        if (
            float(next_cfg.q_min) == float(cfg_current.q_min)
            and float(next_cfg.q_max) == float(cfg_current.q_max)
            and int(next_cfg.n_q) == int(cfg_current.n_q)
        ):
            break
        expansion_directions.append(direction)
        expansion_triggers.append(trigger)
        cfg_current = next_cfg
        if use_incremental_q_expansion:
            left_q, right_q = _incremental_q_strips(current, cfg_current)
            strip_scans: list[ScanResult] = []
            if left_q.size:
                cfg_left = replace(cfg_current, q_min=float(left_q[0]), q_max=float(left_q[-1]), n_q=int(left_q.size))
                left_scan = _run_scan_for_q_vec_with_normal(
                    kT,
                    JA,
                    cfg_left,
                    device,
                    q_edge_margin=q_edge_margin,
                    q_vec=left_q,
                    omega_normal_scalar=float(current.omega_normal_scalar),
                )
                strip_scans.append(left_scan)
                q_expansion_left_runtime_sec += float(left_scan.scan_runtime_sec)
                added_left_q_points_evaluated += int(left_scan.q_points_evaluated)
                incremental_q_grid_evaluations += int(left_scan.estimated_grid_evaluations)
            if right_q.size:
                cfg_right = replace(cfg_current, q_min=float(right_q[0]), q_max=float(right_q[-1]), n_q=int(right_q.size))
                right_scan = _run_scan_for_q_vec_with_normal(
                    kT,
                    JA,
                    cfg_right,
                    device,
                    q_edge_margin=q_edge_margin,
                    q_vec=right_q,
                    omega_normal_scalar=float(current.omega_normal_scalar),
                )
                strip_scans.append(right_scan)
                q_expansion_right_runtime_sec += float(right_scan.scan_runtime_sec)
                added_right_q_points_evaluated += int(right_scan.q_points_evaluated)
                incremental_q_grid_evaluations += int(right_scan.estimated_grid_evaluations)
            if strip_scans:
                merge_t0 = time.perf_counter()
                merged_cache = _merge_q_scan_caches([_scan_to_cache(current, _lvl)] + [_scan_to_cache(s, _lvl + 1) for s in strip_scans])
                merged_scan = _cache_to_scan(merged_cache, q_edge_margin=q_edge_margin)
                merge_cache_runtime_sec += float(time.perf_counter() - merge_t0)
                scans.append(merged_scan)
                incremental_expansion_used = 1
            else:
                fallback_t0 = time.perf_counter()
                fallback_scan = _run_scan_with_normal(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
                fallback_full_rescan_runtime_sec += float(time.perf_counter() - fallback_t0)
                fallback_full_rescan_used = 1
                fallback_full_rescan_reason = "no_incremental_strip"
                recomputed_q_points += int(fallback_scan.q_points_evaluated)
                fallback_full_rescan_grid_evaluations += int(fallback_scan.estimated_grid_evaluations)
                scans.append(fallback_scan)
        else:
            full_scan = _run_scan_with_normal(kT, JA, cfg_current, device, q_edge_margin=q_edge_margin)
            q_expansion_runtime_sec += float(full_scan.scan_runtime_sec)
            recomputed_q_points += int(full_scan.q_points_evaluated)
            fallback_full_rescan_grid_evaluations += int(full_scan.estimated_grid_evaluations)
            scans.append(full_scan)

    q_expansion_runtime_sec += float(q_expansion_left_runtime_sec + q_expansion_right_runtime_sec)

    final_scan = scans[-1]
    final_diag = _diagnose_q_window(final_scan, delta_eps=delta_eps, ambiguity_tol=free_energy_ambiguity_tol)

    # Collect minima from final scan and perform local refinement boxes.
    local_minima_t0 = time.perf_counter()
    minima = _build_branch_candidates(final_scan, local_refine_energy_window=local_refine_energy_window)
    raw_minima_count = int(len(minima))
    if enable_basin_clustering and minima:
        coarse_delta = float((float(base_cfg.delta_max) - float(base_cfg.delta_min)) / max(1, int(base_cfg.n_delta) - 1))
        minima = cluster_branch_candidates(
            minima,
            coarse_dq=float(final_scan.dq),
            coarse_dDelta=float(coarse_delta),
            numerical_energy_scale=max(float(local_refine_energy_window), float(free_energy_ambiguity_tol), 1.0e-12),
            delta_eps=float(delta_eps),
            delta_refine_half_width=float(delta_refine_half_width),
            q_cluster_factor=float(basin_q_cluster_factor),
            delta_cluster_factor=float(basin_delta_cluster_factor),
            energy_cluster_factor=float(basin_energy_cluster_factor),
        )
    local_minima_detection_runtime_sec += float(time.perf_counter() - local_minima_t0)
    minima_sorted = sorted(minima, key=lambda r: float(r["DeltaF_local_min"]))
    pruning_window = float(local_refine_energy_window if local_refine_pruning_energy_window is None else local_refine_pruning_energy_window)
    minima_sorted = mark_energy_window_pruning(
        minima_sorted,
        local_refine_energy_window=float(pruning_window),
        delta_eps=float(delta_eps),
        delta_refine_half_width=float(delta_refine_half_width),
        enabled=bool(energy_window_pruning_enabled),
    )
    refine_targets = select_local_refine_targets(
        minima_sorted,
        delta_eps=float(delta_eps),
        delta_refine_half_width=float(delta_refine_half_width),
        max_total_refined_basins=int(max_refined_minima),
        enable_selective_refinement=bool(enable_selective_refinement),
        max_optional_refined_basins=int(max_optional_refined_basins),
        mandatory_basins_can_exceed_cap=bool(mandatory_basins_can_exceed_cap),
        high_risk_overflow_policy=str(high_risk_overflow_policy),
        max_edge_risk_basins=int(max_edge_risk_basins),
        max_delta_near_eps_basins=int(max_delta_near_eps_basins),
        max_near_degenerate_basins=int(max_near_degenerate_basins),
    )
    selected_target_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for row in refine_targets:
        if "basin_id" in row:
            selected_target_by_key[("basin", int(row.get("basin_id", 0)))] = row
        selected_target_by_key[("rank", int(row.get("minimum_rank", 0)))] = row
    point_mandatory_overflow_count = max(
        (int(row.get("mandatory_overflow_count", 0)) for row in refine_targets),
        default=0,
    )
    point_mandatory_overflow = int(point_mandatory_overflow_count > 0)
    for row in minima_sorted:
        key = ("basin", int(row.get("basin_id", 0))) if "basin_id" in row else ("rank", int(row.get("minimum_rank", 0)))
        selected_row = selected_target_by_key.get(key)
        if selected_row is None:
            row["selected_for_refinement"] = 0
            if bool(row.get("mandatory_basin", False)) and str(high_risk_overflow_policy).strip().lower() == "rank_and_cap":
                row["dropped_mandatory_reason"] = "rank_and_cap_overflow"
            elif bool(row.get("pruned_by_energy_window", False)):
                row["dropped_mandatory_reason"] = "ordinary_pruned_by_energy_window"
            else:
                row["dropped_mandatory_reason"] = "ordinary_not_in_optional_topk"
            row["mandatory_overflow_policy"] = str(high_risk_overflow_policy).strip().lower()
            row["mandatory_overflow"] = int(point_mandatory_overflow)
            row["mandatory_overflow_count"] = int(point_mandatory_overflow_count)
            row["rank_and_cap_selection_reason"] = "not_selected"
            continue
        row["selected_for_refinement"] = 1
        row["dropped_mandatory_reason"] = "not_dropped"
        row["mandatory_overflow_policy"] = str(selected_row.get("mandatory_overflow_policy", high_risk_overflow_policy))
        row["mandatory_overflow"] = int(bool(selected_row.get("mandatory_overflow", False)))
        row["mandatory_overflow_count"] = int(selected_row.get("mandatory_overflow_count", 0))
        row["rank_and_cap_selection_reason"] = str(selected_row.get("rank_and_cap_selection_reason", "selected"))
    clustered_basin_count = int(len(minima_sorted))
    basin_clustering_merged_count = int(max(0, raw_minima_count - clustered_basin_count)) if enable_basin_clustering else 0
    energy_window_pruned_count = int(sum(1 for row in minima_sorted if bool(row.get("pruned_by_energy_window", False))))

    refined_rows: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    for row in minima_sorted:
        r = dict(row)
        r["refined_q"] = float(row["q_local_min"])
        r["refined_Delta"] = float(row["Delta_local_min"])
        r["refined_DeltaF"] = float(row["DeltaF_local_min"])
        r["refined_status"] = "coarse"
        r["pruned_reason"] = str(row.get("pruned_reason", "not_pruned"))
        all_candidates.append(r)

    pre_refine_global_deltaf = float(minima_sorted[0]["DeltaF_local_min"]) if minima_sorted else float(final_scan.deltaf_min)
    pre_refine_global_delta = float(minima_sorted[0]["Delta_local_min"]) if minima_sorted else float(final_scan.delta_opt)
    pre_refine_phase = _phase_from_deltaf(
        pre_refine_global_delta,
        pre_refine_global_deltaf,
        delta_eps=delta_eps,
        free_energy_ambiguity_tol=free_energy_ambiguity_tol,
    )

    for row in refine_targets:
        q_c = float(row["q_local_min"])
        d_c = float(row["Delta_local_min"])
        q_lo = max(float(final_scan.q_min), q_c - float(local_refine_q_half_width))
        q_hi = min(float(final_scan.q_max), q_c + float(local_refine_q_half_width))
        d_lo = max(0.0, d_c - float(local_refine_delta_half_width))
        d_hi = min(float(base_cfg.delta_max), d_c + float(local_refine_delta_half_width))
        if q_hi <= q_lo:
            q_hi = min(float(final_scan.q_max), q_lo + max(float(final_scan.dq), 1e-6))
        if d_hi <= d_lo:
            d_hi = min(float(base_cfg.delta_max), d_lo + max(float(delta_eps), 1e-6))
        cfg_local = replace(
            base_cfg,
            q_min=float(q_lo),
            q_max=float(q_hi),
            n_q=max(int(n_q_refined), int(math.ceil((q_hi - q_lo) / max(float(final_scan.dq), 1e-12))) + 1),
            delta_min=float(d_lo),
            delta_max=float(d_hi),
            n_delta=max(3, int(n_delta_refined)),
        )
        local_scan = _run_scan_with_normal(kT, JA, cfg_local, device, q_edge_margin=q_edge_margin)
        local_refinement_runtime_sec += float(local_scan.scan_runtime_sec)
        local_refinement_grid_evaluations += int(local_scan.estimated_grid_evaluations)
        if local_box_records is not None:
            refined_phase = _phase_from_deltaf(
                float(local_scan.delta_opt),
                float(local_scan.deltaf_min),
                delta_eps=delta_eps,
                free_energy_ambiguity_tol=free_energy_ambiguity_tol,
            )
            local_box_records.append(
                {
                    "point_id": int(point_index),
                    "kT": float(kT),
                    "JA": float(JA),
                    "branch_id": int(row["minimum_rank"]),
                    "branch_rank_before_refine": int(row["minimum_rank"]),
                    "grid_index": int(row["grid_index"]),
                    "candidate_type": "global_best" if int(row["minimum_rank"]) == 1 else "local_minimum",
                    "selection_reason": _local_refine_selection_reason(row, delta_eps, delta_refine_half_width),
                    "basin_id": int(row.get("basin_id", row["minimum_rank"])),
                    "cluster_size": int(row.get("cluster_size", 1)),
                    "merged_branch_ids": str(row.get("merged_branch_ids", row.get("minimum_rank", ""))),
                    "basin_risk_flags": str(row.get("basin_risk_flags", "not_annotated")),
                    "basin_selection_reason": _local_refine_selection_reason(row, delta_eps, delta_refine_half_width),
                    "mandatory_basin_reasons": str(row.get("mandatory_basin_reasons", "none")),
                    "basin_has_global_best": int(
                        _truthy_flag(row, "basin_has_global_best") or int(row.get("minimum_rank", 0)) == 1
                    ),
                    "basin_has_edge_risk": int(_truthy_flag(row, "basin_has_edge_risk") or bool(row.get("edge_risk", False))),
                    "basin_has_delta_near_epsilon": int(
                        _truthy_flag(row, "basin_has_delta_near_epsilon")
                        or _delta_near_epsilon(row, delta_eps, delta_refine_half_width)
                    ),
                    "basin_has_near_degenerate": int(
                        _truthy_flag(row, "basin_has_near_degenerate")
                        or (int(row.get("minimum_rank", 0)) != 1 and bool(row.get("within_low_energy_window", False)))
                    ),
                    "basin_has_low_energy_window": int(
                        _truthy_flag(row, "basin_has_low_energy_window") or bool(row.get("within_low_energy_window", False))
                    ),
                    "basin_is_ordinary": int(
                        _truthy_flag(row, "basin_is_ordinary")
                        or not _mandatory_basin_reasons(row, delta_eps, delta_refine_half_width)
                    ),
                    "q_center_before_refine": float(row["q_local_min"]),
                    "Delta_center_before_refine": float(row["Delta_local_min"]),
                    "DeltaF_before_refine": float(row["DeltaF_local_min"]),
                    "energy_above_global_before_refine": float(row["energy_above_global"]),
                    "distance_to_q_edge_before_refine": float(row["distance_to_q_edge"]),
                    "box_q_min": float(q_lo),
                    "box_q_max": float(q_hi),
                    "box_Delta_min": float(d_lo),
                    "box_Delta_max": float(d_hi),
                    "n_q_local": int(cfg_local.n_q),
                    "n_Delta_local": int(cfg_local.n_delta),
                    "local_grid_evaluations": int(local_scan.estimated_grid_evaluations),
                    "box_runtime_sec": float(local_scan.scan_runtime_sec),
                    "refined_q": float(local_scan.q_opt),
                    "refined_Delta": float(local_scan.delta_opt),
                    "refined_DeltaF": float(local_scan.deltaf_min),
                    "refined_status": "refined_box",
                    "changed_global_minimum": int(float(local_scan.deltaf_min) < pre_refine_global_deltaf - float(free_energy_ambiguity_tol)),
                    "changed_phase_label": int(refined_phase != pre_refine_phase),
                    "near_degenerate_after_refine": int(abs(float(local_scan.deltaf_min) - pre_refine_global_deltaf) <= float(local_refine_energy_window)),
                    "q_window_level": int(len(expansion_directions)),
                    "incremental_qexpansion_used": int(incremental_expansion_used),
                    "reused_from_previous_scan": 0,
                    "pruned_reason": "not_pruned_stage_1",
                }
            )
        refined_rows.append(
            {
                "minimum_rank": int(row["minimum_rank"]),
                "grid_index": int(row["grid_index"]),
                "q_local_min": float(row["q_local_min"]),
                "Delta_local_min": float(row["Delta_local_min"]),
                "DeltaF_local_min": float(row["DeltaF_local_min"]),
                "energy_above_global": float(row["energy_above_global"]),
                "distance_to_q_edge": float(row["distance_to_q_edge"]),
                "within_low_energy_window": bool(row["within_low_energy_window"]),
                "edge_risk": bool(row["edge_risk"]),
                "basin_id": int(row.get("basin_id", row["minimum_rank"])),
                "cluster_size": int(row.get("cluster_size", 1)),
                "merged_branch_ids": str(row.get("merged_branch_ids", row.get("minimum_rank", ""))),
                "cluster_reason": str(row.get("cluster_reason", "not_clustered")),
                "basin_risk_flags": str(row.get("basin_risk_flags", "not_annotated")),
                "basin_has_global_best": int(row.get("basin_has_global_best", int(int(row.get("minimum_rank", 0)) == 1))),
                "basin_has_edge_risk": int(row.get("basin_has_edge_risk", int(bool(row.get("edge_risk", False))))),
                "basin_has_delta_near_epsilon": int(row.get("basin_has_delta_near_epsilon", 0)),
                "basin_has_near_degenerate": int(row.get("basin_has_near_degenerate", 0)),
                "basin_has_low_energy_window": int(row.get("basin_has_low_energy_window", int(bool(row.get("within_low_energy_window", False))))),
                "basin_is_ordinary": int(row.get("basin_is_ordinary", 0)),
                "basin_global_best_candidate_count": int(row.get("basin_global_best_candidate_count", int(int(row.get("minimum_rank", 0)) == 1))),
                "basin_edge_risk_candidate_count": int(row.get("basin_edge_risk_candidate_count", int(bool(row.get("edge_risk", False))))),
                "basin_delta_near_epsilon_candidate_count": int(row.get("basin_delta_near_epsilon_candidate_count", 0)),
                "basin_near_degenerate_candidate_count": int(row.get("basin_near_degenerate_candidate_count", 0)),
                "basin_low_energy_window_candidate_count": int(row.get("basin_low_energy_window_candidate_count", int(bool(row.get("within_low_energy_window", False))))),
                "mandatory_basin_reasons": str(row.get("mandatory_basin_reasons", "none")),
                "basin_selection_reason": str(row.get("basin_selection_reason", "selected")),
                "selected_for_refinement": int(row.get("selected_for_refinement", 1)),
                "dropped_mandatory_reason": str(row.get("dropped_mandatory_reason", "not_dropped")),
                "mandatory_overflow_policy": str(row.get("mandatory_overflow_policy", high_risk_overflow_policy)),
                "mandatory_overflow": int(bool(row.get("mandatory_overflow", False))),
                "mandatory_overflow_count": int(row.get("mandatory_overflow_count", 0)),
                "rank_and_cap_selection_reason": str(row.get("rank_and_cap_selection_reason", "selected")),
                "pruned_reason": str(row.get("pruned_reason", "not_pruned")),
                "basin_q_width": float(row.get("basin_q_width", 0.0)),
                "basin_Delta_width": float(row.get("basin_Delta_width", 0.0)),
                "basin_energy_span": float(row.get("basin_energy_span", 0.0)),
                "basin_curvature_proxy": float(row.get("basin_curvature_proxy", 0.0)),
                "refined_q": float(local_scan.q_opt),
                "refined_Delta": float(local_scan.delta_opt),
                "refined_DeltaF": float(local_scan.deltaf_min),
                "refined_status": "refined_box",
            }
        )

    rank_to_best: dict[int, dict[str, object]] = {}
    for row in all_candidates:
        rank_to_best[int(row["minimum_rank"])] = row
    for rr in refined_rows:
        rank = int(rr["minimum_rank"])
        if rank not in rank_to_best or float(rr["refined_DeltaF"]) < float(rank_to_best[rank].get("refined_DeltaF", np.inf)):
            rank_to_best[rank] = rr
    candidates_final = list(rank_to_best.values())
    if not candidates_final:
        candidates_final = [
            {
                "minimum_rank": 1,
                "q_local_min": float(final_scan.q_opt),
                "Delta_local_min": float(final_scan.delta_opt),
                "DeltaF_local_min": float(final_scan.deltaf_min),
                "energy_above_global": 0.0,
                "distance_to_q_edge": float(final_scan.q_edge_distance),
                "refined_q": float(final_scan.q_opt),
                "refined_Delta": float(final_scan.delta_opt),
                "refined_DeltaF": float(final_scan.deltaf_min),
                "refined_status": "fallback",
            }
        ]
    candidates_final = sorted(candidates_final, key=lambda r: float(r.get("refined_DeltaF", np.inf)))
    best = candidates_final[0]
    best_deltaf = float(best.get("refined_DeltaF", np.nan))
    best_delta = float(best.get("refined_Delta", np.nan))
    best_q = float(best.get("refined_q", np.nan))
    near_deg = [r for r in candidates_final if np.isfinite(best_deltaf) and np.isfinite(float(r.get("refined_DeltaF", np.nan))) and abs(float(r.get("refined_DeltaF")) - best_deltaf) <= float(local_refine_energy_window)]
    near_degenerate_branch_count = max(0, len(near_deg) - 1)

    # Delta guardrail for tolerance-sensitive points.
    delta_refinement_triggered = int(
        enable_delta_refinement
        and (
            abs(best_deltaf) <= float(free_energy_ambiguity_tol)
            or abs(best_delta - float(delta_eps)) <= float(delta_refine_half_width)
            or best_delta < float(delta_eps)
        )
    )
    delta_refinement_valid = 1
    changed_after_delta_refinement = 0
    boundary_ambiguous = 0
    positive_delta_gap = float("nan")
    positive_delta_checked = 0
    if delta_refinement_triggered:
        cfg_pos = positive_delta_config(base_cfg, delta_eps=delta_eps, n_delta_positive=max(3, int(n_delta_refined)))
        pos_scan = _run_scan_with_normal(kT, JA, cfg_pos, device, q_edge_margin=q_edge_margin)
        delta_refinement_runtime_sec += float(pos_scan.scan_runtime_sec)
        delta_refinement_grid_evaluations += int(pos_scan.estimated_grid_evaluations)
        positive_delta_checked = 1
        positive_delta_gap = float(pos_scan.deltaf_min)
        if best_delta < float(delta_eps):
            if np.isfinite(positive_delta_gap) and positive_delta_gap < -float(positive_delta_gap_tol) and pos_scan.delta_opt > float(delta_eps):
                best_deltaf = float(pos_scan.deltaf_min)
                best_delta = float(pos_scan.delta_opt)
                best_q = float(pos_scan.q_opt)
                changed_after_delta_refinement = 1
            elif np.isfinite(positive_delta_gap) and 0.0 <= positive_delta_gap <= float(positive_delta_gap_tol):
                boundary_ambiguous = 1
                changed_after_delta_refinement = 1
        elif np.isfinite(positive_delta_gap) and abs(positive_delta_gap) <= float(free_energy_ambiguity_tol):
            boundary_ambiguous = 1
            changed_after_delta_refinement = 1

    # Final phase using unchanged thermodynamic criterion.
    if np.isfinite(best_deltaf) and np.isfinite(best_delta) and (best_deltaf < -float(free_energy_ambiguity_tol)) and (best_delta > float(delta_eps)):
        phase_candidate = PHASE_SUPERCONDUCTING
    elif np.isfinite(best_delta) and best_delta < float(delta_eps):
        phase_candidate = PHASE_NORMAL
    elif np.isfinite(best_deltaf) and abs(best_deltaf) <= float(free_energy_ambiguity_tol):
        phase_candidate = PHASE_AMBIGUOUS
        boundary_ambiguous = 1
    else:
        phase_candidate = PHASE_NORMAL

    q_unresolved = int(
        phase_candidate == PHASE_SUPERCONDUCTING
        and np.isfinite(best_q)
        and min(abs(best_q - float(final_scan.q_min)), abs(float(final_scan.q_max) - best_q)) <= float(final_scan.q_edge_margin)
        and len(expansion_directions) >= int(max_q_expansion_levels)
    )
    q_window_coverage_valid = int(not q_unresolved)
    expanded_window_found_lower_branch = int(
        len(scans) > 1
        and np.isfinite(initial_scan.deltaf_min)
        and np.isfinite(best_deltaf)
        and (best_deltaf < initial_scan.deltaf_min - float(free_energy_ambiguity_tol))
    )
    phase_initial = (
        "superconducting" if (np.isfinite(initial_scan.deltaf_min) and initial_scan.deltaf_min < -float(free_energy_ambiguity_tol) and initial_scan.delta_opt > float(delta_eps))
        else ("ambiguous" if (np.isfinite(initial_scan.deltaf_min) and abs(initial_scan.deltaf_min) <= float(free_energy_ambiguity_tol)) else "normal")
    )
    phase_final_name = _phase_name(phase_candidate)
    phase_changed_after_q_expansion = int(len(scans) > 1 and phase_initial != phase_final_name)

    q_status = Q_NOT_APPLICABLE if phase_candidate != PHASE_SUPERCONDUCTING else (Q_UNRESOLVED if q_unresolved else (Q_EXPANDED_CONFIRMED if len(scans) > 1 else Q_ACTIVE))
    q_edge_hit_final = int(phase_candidate == PHASE_SUPERCONDUCTING and np.isfinite(best_q) and min(abs(best_q - float(final_scan.q_min)), abs(float(final_scan.q_max) - best_q)) <= float(final_scan.q_edge_margin))
    delta_status = DELTA_BOUNDARY_AMBIGUOUS if boundary_ambiguous else DELTA_STABLE
    delta_unresolved = int(boundary_ambiguous and phase_candidate != PHASE_NORMAL and not delta_refinement_valid)
    status_code = 0
    unresolved_reason = ""
    if q_unresolved:
        status_code |= STATUS_Q_EDGE_UNRESOLVED
    if delta_unresolved:
        status_code |= STATUS_DELTA_BOUNDARY_UNRESOLVED
    if not np.isfinite([best_q, best_delta, best_deltaf]).all():
        status_code |= STATUS_NONFINITE_OUTPUT
        unresolved_reason = "nonfinite_output"
    if status_code != 0 and not unresolved_reason:
        unresolved_reason = _status_name(status_code)
    stable_normal = bool(
        phase_candidate == PHASE_NORMAL
        and np.isfinite(positive_delta_gap)
        and positive_delta_gap > float(positive_delta_gap_tol)
    )
    boundary_band_normal = bool(
        phase_candidate == PHASE_NORMAL
        and boundary_ambiguous
        and np.isfinite(positive_delta_gap)
        and 0.0 <= positive_delta_gap <= float(positive_delta_gap_tol)
    )
    trusted_exact = int(status_code == 0 and (not boundary_ambiguous or stable_normal))
    training_eligible_exact = int(status_code == 0 and (trusted_exact or boundary_band_normal))
    if status_code & STATUS_NONFINITE_OUTPUT:
        confidence_state = "solver_failed"
    elif q_unresolved:
        confidence_state = "coverage_unresolved"
    elif boundary_ambiguous:
        confidence_state = "boundary_ambiguous"
    else:
        confidence_state = "trusted"
    rerun_required = int(not bool(training_eligible_exact))

    # Current response on final q-grid for metadata compatibility.
    q_vec_final = final_scan.q_vec
    j_q = compute_current_from_omega(final_scan.omega_sc_q.reshape(1, 1, -1), q_vec_final)[0, 0]
    iq_opt = int(np.argmin(np.abs(q_vec_final - best_q))) if np.isfinite(best_q) else int(final_scan.q_index)
    eta, ic_plus, ic_minus = find_eta_from_jq(j_q, q_vec_final, iq_opt)

    branch_path = "N/A"
    if branch_dir is not None:
        branch_dir.mkdir(parents=True, exist_ok=True)
        branch_file = branch_dir / f"point{int(point_index):04d}_branches.csv"
        _write_branch_candidates_csv(branch_file, point_id=int(point_index), kT=kT, JA=JA, rows=candidates_final)
        branch_path = str(branch_file.resolve())

    direction_str = ";".join(expansion_directions) if expansion_directions else "none"
    trigger_str = ";".join(expansion_triggers) if expansion_triggers else "none"
    initial_dq = float((initial_scan.q_max - initial_scan.q_min) / max(1, int(initial_scan.n_q) - 1))
    final_dq = float((final_scan.q_max - final_scan.q_min) / max(1, int(final_scan.n_q) - 1))
    total_q_points_evaluated = int(base_q_points_evaluated + added_left_q_points_evaluated + added_right_q_points_evaluated + recomputed_q_points)
    total_estimated_grid_evaluations = int(
        base_grid_evaluations
        + incremental_q_grid_evaluations
        + fallback_full_rescan_grid_evaluations
        + delta_refinement_grid_evaluations
        + local_refinement_grid_evaluations
    )
    point_total_runtime_sec = float(time.perf_counter() - point_t0)
    accounted_runtime = float(
        base_scan_runtime_sec
        + q_expansion_runtime_sec
        + delta_refinement_runtime_sec
        + local_refinement_runtime_sec
        + merge_cache_runtime_sec
        + local_minima_detection_runtime_sec
        + fallback_full_rescan_runtime_sec
    )
    other_runtime_sec = float(max(0.0, point_total_runtime_sec - accounted_runtime))
    return ConfirmedPoint(
        kT=float(kT),
        JA=float(JA),
        eta=float(eta),
        q_opt=float(best_q),
        delta_opt=float(best_delta),
        ic_plus=float(ic_plus),
        ic_minus=float(ic_minus),
        phase_candidate=int(phase_candidate),
        q_status=int(q_status),
        q_min=float(final_scan.q_min),
        q_max=float(final_scan.q_max),
        n_q=int(final_scan.n_q),
        q_index=int(iq_opt),
        q_edge_distance=float(min(abs(best_q - final_scan.q_min), abs(final_scan.q_max - best_q)) if np.isfinite(best_q) else np.nan),
        q_edge_hit=int(q_edge_hit_final),
        q_refinement_level=int(len(expansion_directions)),
        q_expanded=int(len(expansion_directions) > 0),
        q_unresolved=int(q_unresolved),
        delta_status=int(delta_status),
        delta_min=float(base_cfg.delta_min),
        delta_max=float(base_cfg.delta_max),
        n_delta=int(base_cfg.n_delta),
        n_delta_refined=int(n_delta_refined if delta_refinement_triggered else 0),
        delta_refinement_level=int(delta_refinement_triggered),
        delta_boundary_ambiguous=int(boundary_ambiguous),
        delta_refined=int(delta_refinement_triggered),
        delta_unresolved=int(delta_unresolved),
        free_energy_gap_to_normal=float(best_deltaf),
        positive_delta_gap=float(positive_delta_gap),
        positive_delta_checked=int(positive_delta_checked),
        exact_status_code=int(status_code),
        exact_status_name=_status_name(int(status_code)),
        trusted_exact=int(trusted_exact),
        confidence_state=str(confidence_state),
        training_eligible_exact=int(training_eligible_exact),
        rerun_required=int(rerun_required),
        oracle_mode="robust_al",
        search_mode="adaptive_box",
        initial_q_min=float(initial_scan.q_min),
        initial_q_max=float(initial_scan.q_max),
        final_q_min=float(final_scan.q_min),
        final_q_max=float(final_scan.q_max),
        initial_n_q=int(initial_scan.n_q),
        final_n_q=int(final_scan.n_q),
        initial_dq=float(initial_dq),
        final_dq=float(final_dq),
        q_expansion_count=int(len(expansion_directions)),
        q_expansion_directions=direction_str,
        q_expansion_trigger=trigger_str,
        q_window_coverage_valid=int(q_window_coverage_valid),
        q_window_unresolved=int(q_unresolved),
        qopt_edge_hit_initial=int(initial_scan.qopt_edge_hit),
        qopt_edge_hit_final=int(q_edge_hit_final),
        edge_risk_left_initial=int(bool(initial_diag.get("left_edge_low_energy")) or bool(initial_diag.get("left_edge_descent_outward"))),
        edge_risk_right_initial=int(bool(initial_diag.get("right_edge_low_energy")) or bool(initial_diag.get("right_edge_descent_outward"))),
        edge_risk_left_final=int(bool(final_diag.get("left_edge_low_energy")) or bool(final_diag.get("left_edge_descent_outward"))),
        edge_risk_right_final=int(bool(final_diag.get("right_edge_low_energy")) or bool(final_diag.get("right_edge_descent_outward"))),
        expanded_window_found_lower_branch=int(expanded_window_found_lower_branch),
        phase_changed_after_q_expansion=int(phase_changed_after_q_expansion),
        local_minima_count=int(raw_minima_count),
        refined_local_minima_count=int(len(refined_rows)),
        near_degenerate_branch_count=int(near_degenerate_branch_count),
        selected_minimum_rank=int(best.get("minimum_rank", 1)),
        branch_candidates_path=branch_path,
        delta_refinement_triggered=int(delta_refinement_triggered),
        delta_refinement_valid=int(delta_refinement_valid),
        boundary_ambiguous=int(boundary_ambiguous),
        changed_after_delta_refinement=int(changed_after_delta_refinement),
        unresolved_reason=(unresolved_reason if unresolved_reason else "N/A"),
        point_total_runtime_sec=float(point_total_runtime_sec),
        base_scan_runtime_sec=float(base_scan_runtime_sec),
        q_expansion_runtime_sec=float(q_expansion_runtime_sec + fallback_full_rescan_runtime_sec),
        q_expansion_left_runtime_sec=float(q_expansion_left_runtime_sec),
        q_expansion_right_runtime_sec=float(q_expansion_right_runtime_sec),
        delta_refinement_runtime_sec=float(delta_refinement_runtime_sec),
        local_refinement_runtime_sec=float(local_refinement_runtime_sec),
        merge_cache_runtime_sec=float(merge_cache_runtime_sec),
        local_minima_detection_runtime_sec=float(local_minima_detection_runtime_sec),
        fallback_full_rescan_runtime_sec=float(fallback_full_rescan_runtime_sec),
        other_runtime_sec=float(other_runtime_sec),
        base_q_points_evaluated=int(base_q_points_evaluated),
        added_left_q_points_evaluated=int(added_left_q_points_evaluated),
        added_right_q_points_evaluated=int(added_right_q_points_evaluated),
        recomputed_q_points=int(recomputed_q_points),
        total_q_points_evaluated=int(total_q_points_evaluated),
        base_grid_evaluations=int(base_grid_evaluations),
        incremental_q_grid_evaluations=int(incremental_q_grid_evaluations),
        fallback_full_rescan_grid_evaluations=int(fallback_full_rescan_grid_evaluations),
        delta_refinement_grid_evaluations=int(delta_refinement_grid_evaluations),
        local_refinement_grid_evaluations=int(local_refinement_grid_evaluations),
        total_estimated_grid_evaluations=int(total_estimated_grid_evaluations),
        incremental_expansion_used=int(incremental_expansion_used),
        fallback_full_rescan_used=int(fallback_full_rescan_used),
        fallback_full_rescan_reason=str(fallback_full_rescan_reason),
        local_minima_detected_count=int(raw_minima_count),
        clustered_basin_count=int(clustered_basin_count),
        selected_refine_target_count=int(len(refine_targets)),
        basin_clustering_enabled=int(enable_basin_clustering),
        basin_clustering_merged_count=int(basin_clustering_merged_count),
        energy_window_pruning_enabled=int(energy_window_pruning_enabled),
        energy_window_pruned_count=int(energy_window_pruned_count),
        local_boxes_refined_count=int(len(refined_rows)),
        local_refinement_reused_count=0,
    )


def _confirm_one_point(
    kT: float,
    JA: float,
    base_cfg: EtaPhaseConfig,
    device: torch.device,
    delta_eps: float,
    delta_boundary_margin: float,
    q_edge_margin: float | None,
    free_energy_ambiguity_tol: float,
    positive_delta_gap_tol: float,
    enable_q_expansion: bool,
    q_expand_factor: float,
    q_expand_pad_steps: int,
    q_max_abs: float,
    max_q_refinements: int,
    enable_delta_refinement: bool,
    delta_refine_half_width: float,
    n_delta_refined: int,
    max_delta_refinements: int,
    allow_ambiguous_output: bool,
    oracle_mode: str,
    branch_dir: Path | None,
    point_index: int,
    enable_incremental_q_expansion: bool = False,
    local_box_records: list[dict[str, object]] | None = None,
    enable_basin_clustering: bool = False,
    basin_q_cluster_factor: float = 1.5,
    basin_delta_cluster_factor: float = 1.5,
    basin_energy_cluster_factor: float = 1.0,
    max_refined_minima: int = 6,
    enable_selective_refinement: bool = False,
    max_optional_refined_basins: int = 3,
    mandatory_basins_can_exceed_cap: bool = True,
    high_risk_overflow_policy: str = "keep_all",
    max_edge_risk_basins: int = 1,
    max_delta_near_eps_basins: int = 2,
    max_near_degenerate_basins: int = 2,
    energy_window_pruning_enabled: bool = False,
    local_refine_pruning_energy_window: float | None = None,
) -> ConfirmedPoint:
    mode_l = str(oracle_mode).lower()
    if mode_l in {"robust_al", "adaptive_box", "robust", "robust_incremental"}:
        return _confirm_one_point_robust(
            kT=kT,
            JA=JA,
            base_cfg=base_cfg,
            device=device,
            delta_eps=delta_eps,
            q_eps=1.0e-2,
            q_edge_margin=q_edge_margin,
            free_energy_ambiguity_tol=free_energy_ambiguity_tol,
            positive_delta_gap_tol=positive_delta_gap_tol,
            q_max_abs=q_max_abs,
            max_q_expansion_levels=max_q_refinements,
            local_refine_energy_window=max(1.0e-5, 5.0 * free_energy_ambiguity_tol),
            max_refined_minima=int(max_refined_minima),
            local_refine_q_half_width=max(4.0 * abs((base_cfg.q_max - base_cfg.q_min) / max(1, base_cfg.n_q - 1)), 0.03),
            local_refine_delta_half_width=max(delta_refine_half_width, 0.01),
            n_q_refined=max(2 * int(base_cfg.n_q), 800),
            n_delta_refined=max(int(n_delta_refined), 600),
            enable_delta_refinement=enable_delta_refinement,
            delta_refine_half_width=delta_refine_half_width,
            max_delta_refinements=max_delta_refinements,
            branch_dir=branch_dir,
            point_index=point_index,
            use_incremental_q_expansion=bool(enable_incremental_q_expansion or mode_l == "robust_incremental"),
            local_box_records=local_box_records,
            enable_basin_clustering=bool(enable_basin_clustering),
            basin_q_cluster_factor=float(basin_q_cluster_factor),
            basin_delta_cluster_factor=float(basin_delta_cluster_factor),
            basin_energy_cluster_factor=float(basin_energy_cluster_factor),
            enable_selective_refinement=bool(enable_selective_refinement),
            max_optional_refined_basins=int(max_optional_refined_basins),
            mandatory_basins_can_exceed_cap=bool(mandatory_basins_can_exceed_cap),
            high_risk_overflow_policy=str(high_risk_overflow_policy),
            max_edge_risk_basins=int(max_edge_risk_basins),
            max_delta_near_eps_basins=int(max_delta_near_eps_basins),
            max_near_degenerate_basins=int(max_near_degenerate_basins),
            energy_window_pruning_enabled=bool(energy_window_pruning_enabled),
            local_refine_pruning_energy_window=local_refine_pruning_energy_window,
        )
    return _confirm_one_point_legacy(
        kT=kT,
        JA=JA,
        base_cfg=base_cfg,
        device=device,
        delta_eps=delta_eps,
        delta_boundary_margin=delta_boundary_margin,
        q_edge_margin=q_edge_margin,
        free_energy_ambiguity_tol=free_energy_ambiguity_tol,
        positive_delta_gap_tol=positive_delta_gap_tol,
        enable_q_expansion=enable_q_expansion,
        q_expand_factor=q_expand_factor,
        q_expand_pad_steps=q_expand_pad_steps,
        q_max_abs=q_max_abs,
        max_q_refinements=max_q_refinements,
        enable_delta_refinement=enable_delta_refinement,
        delta_refine_half_width=delta_refine_half_width,
        n_delta_refined=n_delta_refined,
        max_delta_refinements=max_delta_refinements,
        allow_ambiguous_output=allow_ambiguous_output,
    )


def _rows_to_result(rows: list[ConfirmedPoint]) -> OracleResult:
    data = {field: np.asarray([getattr(r, field) for r in rows]) for field in ConfirmedPoint.__dataclass_fields__}
    int_fields = {
        "phase_candidate",
        "q_status",
        "n_q",
        "q_index",
        "q_edge_hit",
        "q_refinement_level",
        "q_expanded",
        "q_unresolved",
        "delta_status",
        "n_delta",
        "n_delta_refined",
        "delta_refinement_level",
        "delta_boundary_ambiguous",
        "delta_refined",
        "delta_unresolved",
        "positive_delta_checked",
        "exact_status_code",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "initial_n_q",
        "final_n_q",
        "q_expansion_count",
        "q_window_coverage_valid",
        "q_window_unresolved",
        "qopt_edge_hit_initial",
        "qopt_edge_hit_final",
        "edge_risk_left_initial",
        "edge_risk_right_initial",
        "edge_risk_left_final",
        "edge_risk_right_final",
        "expanded_window_found_lower_branch",
        "phase_changed_after_q_expansion",
        "local_minima_count",
        "refined_local_minima_count",
        "near_degenerate_branch_count",
        "selected_minimum_rank",
        "delta_refinement_triggered",
        "delta_refinement_valid",
        "boundary_ambiguous",
        "changed_after_delta_refinement",
        "base_q_points_evaluated",
        "added_left_q_points_evaluated",
        "added_right_q_points_evaluated",
        "recomputed_q_points",
        "total_q_points_evaluated",
        "base_grid_evaluations",
        "incremental_q_grid_evaluations",
        "fallback_full_rescan_grid_evaluations",
        "delta_refinement_grid_evaluations",
        "local_refinement_grid_evaluations",
        "total_estimated_grid_evaluations",
        "incremental_expansion_used",
        "fallback_full_rescan_used",
        "local_minima_detected_count",
        "clustered_basin_count",
        "selected_refine_target_count",
        "basin_clustering_enabled",
        "basin_clustering_merged_count",
        "energy_window_pruning_enabled",
        "energy_window_pruned_count",
        "local_boxes_refined_count",
        "local_refinement_reused_count",
        "topology_enabled",
        "topology_applicable",
        "topology_pending",
        "topology_label_code",
        "topology_z2",
        "topology_spectral_status_code",
        "topology_trusted",
        "topology_gap_nk",
        "topology_gap_backend_code",
        "topology_error_code",
    }
    str_fields = {
        "exact_status_name",
        "confidence_state",
        "oracle_mode",
        "search_mode",
        "q_expansion_directions",
        "q_expansion_trigger",
        "branch_candidates_path",
        "unresolved_reason",
        "fallback_full_rescan_reason",
    }
    float_fields = set(ConfirmedPoint.__dataclass_fields__) - int_fields - str_fields
    for key in int_fields:
        data[key] = data[key].astype(np.int64)
    for key in float_fields:
        data[key] = data[key].astype(np.float64)
    for key in str_fields:
        data[key] = data[key].astype(str)
    return OracleResult(**data)


def evaluate_points(
    points: np.ndarray,
    cfg: EtaPhaseConfig | None = None,
    device: str | torch.device | None = None,
    save_every: int = 1,
    output_file: Path | None = None,
    delta_eps: float = 1e-3,
    delta_boundary_margin: float = 2e-2,
    q_edge_margin: float | None = None,
    free_energy_ambiguity_tol: float = 1e-6,
    positive_delta_gap_tol: float = 1e-8,
    enable_q_expansion: bool = False,
    q_expand_factor: float = 1.5,
    q_expand_pad_steps: int = 50,
    q_max_abs: float = math.pi,
    max_q_refinements: int = 3,
    enable_delta_refinement: bool = False,
    delta_refine_half_width: float = 0.03,
    n_delta_refined: int = 300,
    max_delta_refinements: int = 2,
    allow_ambiguous_output: bool = False,
    oracle_mode: str = "legacy",
    branch_dir: Path | None = None,
    enable_incremental_q_expansion: bool = False,
    enable_local_box_instrumentation: bool = False,
    local_box_output_file: Path | None = None,
    enable_basin_clustering: bool = False,
    basin_q_cluster_factor: float = 1.5,
    basin_delta_cluster_factor: float = 1.5,
    basin_energy_cluster_factor: float = 1.0,
    max_refined_minima: int = 6,
    enable_selective_refinement: bool = False,
    max_optional_refined_basins: int = 3,
    mandatory_basins_can_exceed_cap: bool = True,
    high_risk_overflow_policy: str = "keep_all",
    max_edge_risk_basins: int = 1,
    max_delta_near_eps_basins: int = 2,
    max_near_degenerate_basins: int = 2,
    energy_window_pruning_enabled: bool = False,
    local_refine_pruning_energy_window: float | None = None,
    enable_topology_classification: bool = False,
    topology_gap_nk: int = 2048,
    topology_gap_backend: str = "cpu",
    topology_gap_tol_rel: float = 1.0e-8,
    topology_gap_tol_abs: float = 0.0,
    topology_gap_k_chunk: int = 512,
) -> OracleResult:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("points must be shape (n, 2) [kT, JA] or shape (n, 3) [kT, JA, mu]")

    base_cfg = cfg if cfg is not None else EtaPhaseConfig()
    device_obj = torch.device(device) if isinstance(device, str) else (device or _device_from_arg(None))
    maybe_set_linalg_backend(base_cfg.scaled())

    rows: list[ConfirmedPoint] = []
    local_box_records: list[dict[str, object]] | None = [] if enable_local_box_instrumentation else None
    topo_backend = str(topology_gap_backend).lower()
    if topo_backend == "auto":
        topo_backend = "gpu" if torch.cuda.is_available() else "cpu"
    if topo_backend not in {"cpu", "gpu"}:
        raise ValueError("topology_gap_backend must be 'cpu', 'gpu', or 'auto'.")
    topo_pf_oracle: TopologyPfaffianOracle | None = None
    topo_gap_oracle: BulkGapOracle | None = None
    if enable_topology_classification:
        topo_params = TopologyModelParams()
        topo_pf_oracle = TopologyPfaffianOracle(topo_params, pf_tol_rel=1.0e-8)
        topo_gap_oracle = BulkGapOracle(
            topo_params,
            backend=topo_backend,  # type: ignore[arg-type]
            point_chunk=1,
            k_chunk=int(topology_gap_k_chunk),
        )

    def flush_partial() -> None:
        if output_file is None:
            return
        result = _rows_to_result(rows)
        payload = result.to_dict()
        payload["completed_points"] = np.asarray([len(rows)], dtype=np.int64)
        np.savez(output_file, **payload)
        if local_box_records is not None and local_box_output_file is not None:
            _write_local_box_records_csv(local_box_output_file, local_box_records)

    for i, row in enumerate(points):
        kT = float(row[0])
        JA = float(row[1])
        mu = float(row[2]) if points.shape[1] >= 3 else float(base_cfg.mu)
        point_cfg = replace(base_cfg, mu=mu)
        point = _confirm_one_point(
                kT,
                JA,
                base_cfg=point_cfg,
                device=device_obj,
                delta_eps=float(delta_eps),
                delta_boundary_margin=float(delta_boundary_margin),
                q_edge_margin=q_edge_margin,
                free_energy_ambiguity_tol=float(free_energy_ambiguity_tol),
                positive_delta_gap_tol=float(positive_delta_gap_tol),
                enable_q_expansion=bool(enable_q_expansion),
                q_expand_factor=float(q_expand_factor),
                q_expand_pad_steps=int(q_expand_pad_steps),
                q_max_abs=float(q_max_abs),
                max_q_refinements=int(max_q_refinements),
                enable_delta_refinement=bool(enable_delta_refinement),
                delta_refine_half_width=float(delta_refine_half_width),
                n_delta_refined=int(n_delta_refined),
                max_delta_refinements=int(max_delta_refinements),
                allow_ambiguous_output=bool(allow_ambiguous_output),
                oracle_mode=str(oracle_mode),
                branch_dir=branch_dir,
                point_index=int(i),
                enable_incremental_q_expansion=bool(enable_incremental_q_expansion),
                local_box_records=local_box_records,
                enable_basin_clustering=bool(enable_basin_clustering),
                basin_q_cluster_factor=float(basin_q_cluster_factor),
                basin_delta_cluster_factor=float(basin_delta_cluster_factor),
                basin_energy_cluster_factor=float(basin_energy_cluster_factor),
                max_refined_minima=int(max_refined_minima),
                enable_selective_refinement=bool(enable_selective_refinement),
                max_optional_refined_basins=int(max_optional_refined_basins),
                mandatory_basins_can_exceed_cap=bool(mandatory_basins_can_exceed_cap),
                high_risk_overflow_policy=str(high_risk_overflow_policy),
                max_edge_risk_basins=int(max_edge_risk_basins),
                max_delta_near_eps_basins=int(max_delta_near_eps_basins),
                max_near_degenerate_basins=int(max_near_degenerate_basins),
                energy_window_pruning_enabled=bool(energy_window_pruning_enabled),
                local_refine_pruning_energy_window=local_refine_pruning_energy_window,
            )
        point.mu = mu
        point = _attach_topology_diagnostics(
            point,
            enabled=bool(enable_topology_classification),
            pf_oracle=topo_pf_oracle,
            gap_oracle=topo_gap_oracle,
            gap_nk=int(topology_gap_nk),
            gap_tol_rel=float(topology_gap_tol_rel),
            gap_tol_abs=float(topology_gap_tol_abs),
            gap_backend=topo_backend,
        )
        rows.append(point)
        if save_every > 0 and output_file is not None and ((i + 1) % save_every == 0):
            flush_partial()

    flush_partial()
    if local_box_records is not None and local_box_output_file is not None:
        _write_local_box_records_csv(local_box_output_file, local_box_records)
    return _rows_to_result(rows)


def _infer_rank_and_world_size(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if rank is not None and world_size is not None:
        return rank, world_size
    if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    return (rank or 0), (world_size or 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact pointwise BdG oracle for active-learning refinement.")
    p.add_argument("--points-file", type=Path, default=None, help="CSV file with columns kT, JA and optional mu.")
    p.add_argument("--output-file", type=Path, default=None, help="Output npz path for shard result.")
    p.add_argument("--device", type=str, default=None, help="Torch device, e.g., cuda:0 or cpu.")
    p.add_argument("--save-every", type=int, default=1, help="Flush partial output every N points.")
    p.add_argument("--delta-eps", type=float, default=1e-3, help="Delta threshold used for boundary metadata.")
    p.add_argument(
        "--delta-boundary-margin",
        type=float,
        default=2e-2,
        help="Mark Delta metadata ambiguous within this distance of delta_eps.",
    )
    p.add_argument(
        "--q-edge-margin",
        type=float,
        default=None,
        help="Mark q_edge_hit within this distance of q_min/q_max. Defaults to two q-grid steps.",
    )
    p.add_argument(
        "--free-energy-ambiguity-tol",
        type=float,
        default=1e-6,
        help="Mark Delta ambiguity when the free-energy gain is this small.",
    )
    p.add_argument(
        "--positive-delta-gap-tol",
        type=float,
        default=1e-8,
        help="Normal-state Delta=0 is ambiguous only if the best positive-Delta gap is within this tolerance.",
    )
    p.add_argument("--enable-q-expansion", action="store_true", help="Expand q window for SC q-edge points.")
    p.add_argument("--q-expand-factor", type=float, default=1.5, help="q-window expansion factor.")
    p.add_argument("--q-expand-pad-steps", type=int, default=50, help="Minimum added q-grid steps on hit side.")
    p.add_argument("--q-max-abs", type=float, default=math.pi, help="Absolute q-window limit.")
    p.add_argument("--max-q-refinements", type=int, default=3, help="Maximum q-window expansion reruns.")
    p.add_argument("--enable-delta-refinement", action="store_true", help="Refine Delta near normal/SC boundary.")
    p.add_argument("--delta-refine-half-width", type=float, default=0.03, help="Local Delta half-width.")
    p.add_argument("--n-delta-refined", type=int, default=300, help="Number of Delta points in refined scan.")
    p.add_argument("--max-delta-refinements", type=int, default=2, help="Maximum Delta refinement reruns.")
    p.add_argument(
        "--allow-ambiguous-output",
        action="store_true",
        help="Write ambiguous outputs without marking them trusted.",
    )
    p.add_argument(
        "--oracle-mode",
        type=str,
        default="legacy",
        choices=["legacy", "robust_al", "robust_incremental"],
        help="Exact-oracle search mode.",
    )
    p.add_argument(
        "--enable-incremental-q-expansion",
        action="store_true",
        help="For robust modes, evaluate only newly exposed q strips after q-window expansion.",
    )
    p.add_argument(
        "--branch-dir",
        type=Path,
        default=None,
        help="Directory for per-point branch candidate CSVs.",
    )
    p.add_argument(
        "--enable-local-box-instrumentation",
        action="store_true",
        help="Write logging-only local-refinement box timing/effectiveness records.",
    )
    p.add_argument(
        "--local-box-output-file",
        type=Path,
        default=None,
        help="Optional CSV path for local box instrumentation records.",
    )
    p.add_argument(
        "--enable-basin-clustering",
        action="store_true",
        help="Cluster duplicate coarse minima into basin representatives before local refinement.",
    )
    p.add_argument("--basin-q-cluster-factor", type=float, default=1.5, help="q clustering tolerance in coarse dq units.")
    p.add_argument(
        "--basin-delta-cluster-factor",
        type=float,
        default=1.5,
        help="Delta clustering tolerance in coarse dDelta units.",
    )
    p.add_argument(
        "--basin-energy-cluster-factor",
        type=float,
        default=1.0,
        help="Energy clustering tolerance in numerical energy-scale units.",
    )
    p.add_argument(
        "--enable-selective-refinement",
        action="store_true",
        help="Refine all mandatory-risk basins plus a capped number of ordinary basins.",
    )
    p.add_argument(
        "--max-refined-minima",
        type=int,
        default=6,
        help="Maximum total local-refinement basins. Defaults to the legacy cap of 6.",
    )
    p.add_argument(
        "--max-optional-refined-basins",
        type=int,
        default=3,
        help="Maximum ordinary non-mandatory basins to refine when selective refinement is enabled.",
    )
    p.add_argument(
        "--mandatory-basins-can-exceed-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow mandatory-risk basins to exceed the total refined-basin cap.",
    )
    p.add_argument(
        "--high-risk-overflow-policy",
        type=str,
        default="keep_all",
        choices=["keep_all", "rank_and_cap"],
        help="Policy for mandatory-risk basins when they exceed the total refined-basin cap.",
    )
    p.add_argument(
        "--max-edge-risk-basins",
        type=int,
        default=1,
        help="Maximum edge-risk basins kept by rank_and_cap overflow policy.",
    )
    p.add_argument(
        "--max-delta-near-eps-basins",
        type=int,
        default=2,
        help="Maximum Delta-near-epsilon basins kept by rank_and_cap overflow policy.",
    )
    p.add_argument(
        "--max-near-degenerate-basins",
        type=int,
        default=2,
        help="Maximum near-degenerate basins kept by rank_and_cap overflow policy.",
    )
    p.add_argument(
        "--energy-window-pruning-enabled",
        action="store_true",
        help="Prune ordinary non-mandatory basins above the local-refinement energy window.",
    )
    p.add_argument(
        "--local-refine-pruning-energy-window",
        type=float,
        default=None,
        help="Optional energy-above-global window for ordinary-basin pruning. Defaults to local_refine_energy_window.",
    )
    p.add_argument(
        "--enable-topology-classification",
        action="store_true",
        help="Attach active-loop Pfaffian and bulk-gap topology diagnostics to exact outputs.",
    )
    p.add_argument(
        "--topology-gap-nk",
        type=int,
        default=2048,
        help="Full-Brillouin-zone k grid size for active-loop topology bulk-gap diagnostics.",
    )
    p.add_argument(
        "--topology-gap-backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="Backend for active-loop topology bulk-gap diagnostics.",
    )
    p.add_argument(
        "--topology-gap-tol-rel",
        type=float,
        default=1.0e-8,
        help="Relative bulk-gap tolerance scaled by the topology energy scale.",
    )
    p.add_argument(
        "--topology-gap-tol-abs",
        type=float,
        default=0.0,
        help="Absolute bulk-gap tolerance floor.",
    )
    p.add_argument(
        "--topology-gap-k-chunk",
        type=int,
        default=512,
        help="k chunk size for active-loop topology bulk-gap scans.",
    )

    p.add_argument("--run-id", type=str, default=None, help="Run id under ML_Phase/active_runs.")
    p.add_argument("--iteration", type=int, default=None, help="Iteration index for default shard paths.")
    p.add_argument("--rank", type=int, default=None, help="Rank index.")
    p.add_argument("--world-size", type=int, default=None, help="Total ranks.")
    p.add_argument("--active-root", type=Path, default=Path("ML_Phase/active_runs"), help="Active runs root.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace, rank: int, world_size: int) -> tuple[Path, Path]:
    if args.points_file is not None and args.output_file is not None:
        return args.points_file, args.output_file
    if args.run_id is None or args.iteration is None:
        raise ValueError("Either --points-file/--output-file or --run-id/--iteration must be provided.")
    iter_dir = args.active_root / args.run_id / f"iter{args.iteration:03d}"
    points_file = iter_dir / f"selected_points_rank{rank:03d}_of{world_size:03d}.csv"
    output_file = iter_dir / f"exact_shard_rank{rank:03d}_of{world_size:03d}.npz"
    return points_file, output_file


def main() -> None:
    args = _parse_args()
    rank, world_size = _infer_rank_and_world_size(args.rank, args.world_size)
    points_file, output_file = _resolve_paths(args, rank, world_size)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    points_df = pd.read_csv(points_file)
    point_columns = ["kT", "JA"] + (["mu"] if "mu" in points_df.columns else [])
    points = points_df[point_columns].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    if args.local_box_output_file is not None:
        local_box_output_file = args.local_box_output_file
    elif args.enable_local_box_instrumentation:
        perf_dir = output_file.parent / "performance"
        local_box_output_file = perf_dir / f"iter{int(args.iteration or 0):03d}_local_box_timing_rank{rank:03d}_of{world_size:03d}.csv"
    else:
        local_box_output_file = None
    result = evaluate_points(
        points=points,
        cfg=EtaPhaseConfig(),
        device=args.device,
        save_every=max(1, int(args.save_every)),
        output_file=output_file,
        delta_eps=float(args.delta_eps),
        delta_boundary_margin=float(args.delta_boundary_margin),
        q_edge_margin=args.q_edge_margin,
        free_energy_ambiguity_tol=float(args.free_energy_ambiguity_tol),
        positive_delta_gap_tol=float(args.positive_delta_gap_tol),
        enable_q_expansion=bool(args.enable_q_expansion),
        q_expand_factor=float(args.q_expand_factor),
        q_expand_pad_steps=int(args.q_expand_pad_steps),
        q_max_abs=float(args.q_max_abs),
        max_q_refinements=int(args.max_q_refinements),
        enable_delta_refinement=bool(args.enable_delta_refinement),
        delta_refine_half_width=float(args.delta_refine_half_width),
        n_delta_refined=int(args.n_delta_refined),
        max_delta_refinements=int(args.max_delta_refinements),
        allow_ambiguous_output=bool(args.allow_ambiguous_output),
        oracle_mode=str(args.oracle_mode),
        branch_dir=(args.branch_dir if args.branch_dir is not None else output_file.parent / "branch_candidates"),
        enable_incremental_q_expansion=bool(args.enable_incremental_q_expansion),
        enable_local_box_instrumentation=bool(args.enable_local_box_instrumentation),
        local_box_output_file=local_box_output_file,
        enable_basin_clustering=bool(args.enable_basin_clustering),
        basin_q_cluster_factor=float(args.basin_q_cluster_factor),
        basin_delta_cluster_factor=float(args.basin_delta_cluster_factor),
        basin_energy_cluster_factor=float(args.basin_energy_cluster_factor),
        max_refined_minima=int(args.max_refined_minima),
        enable_selective_refinement=bool(args.enable_selective_refinement),
        max_optional_refined_basins=int(args.max_optional_refined_basins),
        mandatory_basins_can_exceed_cap=bool(args.mandatory_basins_can_exceed_cap),
        high_risk_overflow_policy=str(args.high_risk_overflow_policy),
        max_edge_risk_basins=int(args.max_edge_risk_basins),
        max_delta_near_eps_basins=int(args.max_delta_near_eps_basins),
        max_near_degenerate_basins=int(args.max_near_degenerate_basins),
        energy_window_pruning_enabled=bool(args.energy_window_pruning_enabled),
        local_refine_pruning_energy_window=args.local_refine_pruning_energy_window,
        enable_topology_classification=bool(args.enable_topology_classification),
        topology_gap_nk=int(args.topology_gap_nk),
        topology_gap_backend=str(args.topology_gap_backend),
        topology_gap_tol_rel=float(args.topology_gap_tol_rel),
        topology_gap_tol_abs=float(args.topology_gap_tol_abs),
        topology_gap_k_chunk=int(args.topology_gap_k_chunk),
    )
    elapsed = time.perf_counter() - t0
    local_box_summary_file = None
    if args.enable_local_box_instrumentation and local_box_output_file is not None:
        records = pd.read_csv(local_box_output_file).to_dict("records") if local_box_output_file.exists() else []
        local_box_summary_file = local_box_output_file.with_name(
            f"iter{int(args.iteration or 0):03d}_local_refinement_summary_rank{rank:03d}_of{world_size:03d}.json"
        )
        _write_local_refinement_summary_json(local_box_summary_file, records, n_points=int(points.shape[0]))

    payload: Dict[str, Any] = result.to_dict()
    payload["rank"] = np.asarray([rank], dtype=np.int64)
    payload["world_size"] = np.asarray([world_size], dtype=np.int64)
    payload["elapsed_sec"] = np.asarray([elapsed], dtype=np.float64)
    payload["hostname"] = np.asarray([socket.gethostname()])
    payload["device"] = np.asarray([str(args.device or _device_from_arg(None))])
    payload["cuda_visible_devices"] = np.asarray([os.environ.get("CUDA_VISIBLE_DEVICES", "")])
    np.savez(output_file, **payload)

    meta_path = output_file.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "points_file": str(points_file),
                "output_file": str(output_file),
                "n_points": int(points.shape[0]),
                "elapsed_sec": elapsed,
                "hostname": socket.gethostname(),
                "device": str(args.device or _device_from_arg(None)),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "q_edge_hit_count": int(np.sum(result.q_edge_hit)),
                "q_expanded_count": int(np.sum(result.q_expanded)),
                "q_unresolved_count": int(np.sum(result.q_unresolved)),
                "delta_refined_count": int(np.sum(result.delta_refined)),
                "delta_unresolved_count": int(np.sum(result.delta_unresolved)),
                "positive_delta_checked_count": int(np.sum(result.positive_delta_checked)),
                "trusted_exact_count": int(np.sum(result.trusted_exact)),
                "delta_boundary_ambiguous_count": int(np.sum(result.delta_boundary_ambiguous)),
                "nonzero_status_count": int(np.sum(result.exact_status_code != 0)),
                "oracle_mode": str(args.oracle_mode),
                "enable_incremental_q_expansion": bool(args.enable_incremental_q_expansion),
                "enable_local_box_instrumentation": bool(args.enable_local_box_instrumentation),
                "enable_basin_clustering": bool(args.enable_basin_clustering),
                "enable_selective_refinement": bool(args.enable_selective_refinement),
                "max_refined_minima": int(args.max_refined_minima),
                "max_optional_refined_basins": int(args.max_optional_refined_basins),
                "mandatory_basins_can_exceed_cap": bool(args.mandatory_basins_can_exceed_cap),
                "high_risk_overflow_policy": str(args.high_risk_overflow_policy),
                "max_edge_risk_basins": int(args.max_edge_risk_basins),
                "max_delta_near_eps_basins": int(args.max_delta_near_eps_basins),
                "max_near_degenerate_basins": int(args.max_near_degenerate_basins),
                "energy_window_pruning_enabled": bool(args.energy_window_pruning_enabled),
                "local_refine_pruning_energy_window": args.local_refine_pruning_energy_window,
                "enable_topology_classification": bool(args.enable_topology_classification),
                "topology_gap_nk": int(args.topology_gap_nk),
                "topology_gap_backend": str(args.topology_gap_backend),
                "topology_gap_tol_rel": float(args.topology_gap_tol_rel),
                "topology_gap_tol_abs": float(args.topology_gap_tol_abs),
                "topology_gap_k_chunk": int(args.topology_gap_k_chunk),
                "topology_applicable_count": int(np.sum(result.topology_applicable)),
                "topology_trusted_count": int(np.sum(result.topology_trusted)),
                "topology_gapless_count": int(np.sum(result.topology_label_code == TOPOLOGY_LABEL_GAPLESS_SC)),
                "topology_topological_count": int(np.sum(result.topology_label_code == TOPOLOGY_LABEL_TOPOLOGICAL)),
                "topology_trivial_count": int(np.sum(result.topology_label_code == TOPOLOGY_LABEL_TRIVIAL)),
                "topology_unresolved_count": int(np.sum(result.topology_label_code == TOPOLOGY_LABEL_UNRESOLVED)),
                "topology_runtime_sec_sum": float(np.sum(result.topology_runtime_sec)),
                "clustered_basin_count_sum": int(np.sum(result.clustered_basin_count)),
                "basin_clustering_merged_count_sum": int(np.sum(result.basin_clustering_merged_count)),
                "selected_refine_target_count_sum": int(np.sum(result.selected_refine_target_count)),
                "energy_window_pruned_count_sum": int(np.sum(result.energy_window_pruned_count)),
                "local_box_output_file": str(local_box_output_file) if local_box_output_file is not None else "N/A",
                "local_box_summary_file": str(local_box_summary_file) if local_box_summary_file is not None else "N/A",
                "point_total_runtime_sec_sum": float(np.sum(result.point_total_runtime_sec)),
                "base_scan_runtime_sec_sum": float(np.sum(result.base_scan_runtime_sec)),
                "q_expansion_runtime_sec_sum": float(np.sum(result.q_expansion_runtime_sec)),
                "delta_refinement_runtime_sec_sum": float(np.sum(result.delta_refinement_runtime_sec)),
                "local_refinement_runtime_sec_sum": float(np.sum(result.local_refinement_runtime_sec)),
                "fallback_full_rescan_runtime_sec_sum": float(np.sum(result.fallback_full_rescan_runtime_sec)),
                "total_q_points_evaluated": int(np.sum(result.total_q_points_evaluated)),
                "total_estimated_grid_evaluations": int(np.sum(result.total_estimated_grid_evaluations)),
                "incremental_expansion_used_count": int(np.sum(result.incremental_expansion_used)),
                "fallback_full_rescan_used_count": int(np.sum(result.fallback_full_rescan_used)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Oracle finished rank {rank}/{world_size} with {points.shape[0]} points in {elapsed:.2f}s")
    print(f"Trusted exact points: {int(np.sum(result.trusted_exact))}/{points.shape[0]}")
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
