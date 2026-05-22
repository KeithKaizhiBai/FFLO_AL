from __future__ import annotations

import argparse
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


@dataclass
class OracleResult:
    kT: np.ndarray
    JA: np.ndarray
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

    def to_dict(self) -> Dict[str, np.ndarray]:
        return asdict(self)


def _device_from_arg(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


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
    }
    float_fields = set(ConfirmedPoint.__dataclass_fields__) - int_fields - {"exact_status_name"}
    for key in int_fields:
        data[key] = data[key].astype(np.int64)
    for key in float_fields:
        data[key] = data[key].astype(np.float64)
    data["exact_status_name"] = data["exact_status_name"].astype(str)
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
) -> OracleResult:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be shape (n, 2) with columns [kT, JA]")

    base_cfg = cfg if cfg is not None else EtaPhaseConfig()
    device_obj = torch.device(device) if isinstance(device, str) else (device or _device_from_arg(None))
    maybe_set_linalg_backend(base_cfg.scaled())

    rows: list[ConfirmedPoint] = []

    def flush_partial() -> None:
        if output_file is None:
            return
        result = _rows_to_result(rows)
        payload = result.to_dict()
        payload["completed_points"] = np.asarray([len(rows)], dtype=np.int64)
        np.savez(output_file, **payload)

    for i, (kT, JA) in enumerate(points):
        rows.append(
            _confirm_one_point(
                float(kT),
                float(JA),
                base_cfg=base_cfg,
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
            )
        )
        if save_every > 0 and output_file is not None and ((i + 1) % save_every == 0):
            flush_partial()

    flush_partial()
    return _rows_to_result(rows)


def _infer_rank_and_world_size(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if rank is not None and world_size is not None:
        return rank, world_size
    if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_COUNT" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    return (rank or 0), (world_size or 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact pointwise BdG oracle for active-learning refinement.")
    p.add_argument("--points-file", type=Path, default=None, help="CSV file with columns kT, JA.")
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
    points = points_df[["kT", "JA"]].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
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
    )
    elapsed = time.perf_counter() - t0

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
