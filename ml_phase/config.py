from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ActiveLearningConfig:
    # Run semantics
    run_mode: str = "discovery"
    candidate_domain_mode: str = "full"
    initialization: str = "random_grid"
    initial_seed_size: int = 512
    batch_size_max: int = 256
    batch_size_min: int = 0
    batch_size_min_before_min_iter: int = 64
    batch_size_min_after_min_iter: int = 0
    selection_mode: str = "stochastic"
    sampling_power: float = 2.0
    sampling_power_start: float = 1.5
    sampling_power_mid: float = 2.5
    sampling_power_end: float = 4.0
    sampling_power_mid_iter: int = 10
    sampling_power_end_iter: int = 30
    sampling_power_schedule: str = "piecewise"
    score_threshold_abs: float = 0.0
    score_threshold_rel: float = 0.0
    active_pool_rule: str = "max_threshold"
    active_pool_quantile: float = 0.90
    active_pool_quantile_schedule: str = "piecewise"
    active_pool_quantile_start: float = 0.90
    active_pool_quantile_mid: float = 0.95
    active_pool_quantile_end: float = 0.98
    active_pool_quantile_mid_iter: int = 10
    active_pool_quantile_end_iter: int = 30
    active_pool_rel_to_p95: float = 0.7
    active_pool_min_quantile: float = 0.70
    active_pool_max_fraction_start: float = 0.20
    active_pool_max_fraction_end: float = 0.05
    active_pool_max_fraction_end_iter: int = 30
    active_selection_min_iterations: int = 5
    allow_underfilled_batch_after_min_iter: bool = True
    random_seed: int = 42
    hidden_ground_truth: str = ""

    # Physical thresholds
    delta_eps: float = 1e-3
    q_eps: float = 1e-2
    eta_strong: float = 0.5
    boundary_margin: float = 0.08
    delta_boundary_margin: float = 2e-2
    q_edge_margin: float = 2e-2
    high_ja_q_risk_start: float = 1.2
    q_window_safe_min: float = -1.0
    q_window_safe_max: float = 0.5
    free_energy_ambiguity_tol: float = 1e-6
    positive_delta_gap_tol: float = 1e-8

    # Candidate space
    kt_min: float = 0.0
    kt_max: float = 0.56
    ja_min: float = 0.0
    ja_max: float = 2.12
    n_kt_candidates: int = 241
    n_ja_candidates: int = 321
    prioritize_kt_max: float = 0.5
    prioritize_ja_max: float = 1.2
    finite_t_band_width: float | None = None

    # Acquisition
    cls_margin_tau: float = 0.2
    w_cls_entropy_inner: float = 0.4
    w_cls_margin_inner: float = 0.6
    w_cls_mix: float = 1.0
    w_reg_phase: float = 0.6
    w_delta_boundary: float = 1.0
    w_q_boundary_sc: float = 0.9
    w_gradient_phase: float = 0.5
    w_q_edge_risk: float = 0.4
    w_extrapolation: float = 0.15
    w_ext_schedule: str = "piecewise"
    w_ext_start: float = 0.15
    w_ext_mid: float = 0.08
    w_ext_end: float = 0.03
    w_ext_mid_iter: int = 10
    w_ext_end_iter: int = 30
    w_eta_response: float = 0.3
    w_gradient_response: float = 0.3
    w_reg_response: float = 0.3
    b_delta_gate_mode: str = "normal_sc_competition"
    q_boundary_gate_mode: str = "psc"
    interior_filter_mode: str = "soft_penalty"
    interior_penalty_start_iter: int = 10
    interior_penalty_early: float = 0.5
    interior_penalty_late: float = 0.1
    p_conf_threshold: float = 0.98
    u_ns_low: float = 0.05
    u_uf_low: float = 0.05
    g_phase_low: float = 0.05
    e_q_low: float = 0.05
    e_ext_low: float = 0.05
    observation_repulsion_length: float = 0.02
    observation_repulsion_floor: float = 0.5
    batch_repulsion_length: float = 0.03
    batch_repulsion_floor: float = 0.01
    exact_duplicate_radius_norm: float = 1e-6
    # Kept as a legacy diagnostic threshold, not as a hard acquisition radius.
    diversity_min_dist: float = 0.0075
    delta_scale: float = 0.04
    q_scale: float = 0.04
    eta_scale: float = 0.15
    exclude_existing_exact: bool = True
    existing_exclusion_decimals: int = 4
    existing_min_dist: float = 0.0075
    recent_selection_cooldown_iters: int = 5
    recent_selection_cooldown_decimals: int = 4
    boundary_band_cooldown_enabled: bool = True
    boundary_band_cooldown_decimals: int = 4

    # Boundary diagnostics
    boundary_refinement_mode: str = "off"
    boundary_kt_bin_width: float = 0.005
    boundary_max_local_spacing: float = 0.035
    boundary_position_tol: float = 0.00375
    boundary_stable_stages: int = 2

    # Model training
    seed: int = 42
    n_ensemble: int = 5
    hidden_dim: int = 64
    reg_epochs: int = 240
    cls_epochs: int = 240
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.15

    # Runtime
    points_per_iter: int = 64
    iterations: int = 5
    dry_run: bool = True
    mode: str = "local"
    world_size: int = 1
    partition_strategy: str = "round_robin"
    enable_early_stop: bool = True
    min_new_points_per_iter: int = 8
    max_low_append_iters: int = 2

    # Paths
    output_root: str = "ML_Phase"
    warm_start_output_name: str = "warm_start_dataset"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ActiveLearningConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in raw.items() if k in fields}
        return cls(**payload)

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)

    @property
    def datasets_dir(self) -> Path:
        return self.output_root_path / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.output_root_path / "models"

    @property
    def active_runs_dir(self) -> Path:
        return self.output_root_path / "active_runs"

    @property
    def figures_dir(self) -> Path:
        return self.output_root_path / "figures"

    @property
    def reports_dir(self) -> Path:
        return self.output_root_path / "reports"

    @property
    def hpc_jobs_dir(self) -> Path:
        return self.output_root_path / "hpc_jobs"


def ensure_output_dirs(cfg: ActiveLearningConfig) -> None:
    for p in (
        cfg.output_root_path,
        cfg.datasets_dir,
        cfg.models_dir,
        cfg.active_runs_dir,
        cfg.figures_dir,
        cfg.reports_dir,
        cfg.hpc_jobs_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)


def validate_active_learning_config(cfg: ActiveLearningConfig) -> ActiveLearningConfig:
    run_mode = str(cfg.run_mode)
    domain_mode = str(cfg.candidate_domain_mode)
    selection_mode = str(cfg.selection_mode)
    initialization = str(cfg.initialization)

    if run_mode not in {"discovery", "refinement"}:
        raise ValueError("run_mode must be 'discovery' or 'refinement'.")
    if domain_mode not in {"full", "prior_band"}:
        raise ValueError("candidate_domain_mode must be 'full' or 'prior_band'.")
    if selection_mode not in {"topk", "stochastic"}:
        raise ValueError("selection_mode must be 'topk' or 'stochastic'.")
    if initialization != "random_grid":
        raise ValueError("Only initialization='random_grid' is currently supported.")
    if int(cfg.initial_seed_size) < 0:
        raise ValueError("initial_seed_size must be non-negative.")
    if int(cfg.batch_size_max) < 0:
        raise ValueError("batch_size_max must be non-negative.")
    if int(cfg.batch_size_min) < 0:
        raise ValueError("batch_size_min must be non-negative.")
    if int(cfg.batch_size_min) > int(cfg.batch_size_max):
        raise ValueError("batch_size_min cannot exceed batch_size_max.")
    if int(cfg.batch_size_min_before_min_iter) < 0 or int(cfg.batch_size_min_after_min_iter) < 0:
        raise ValueError("adaptive batch minimums must be non-negative.")
    if float(cfg.sampling_power) <= 0.0:
        raise ValueError("sampling_power must be positive.")
    if (
        float(cfg.sampling_power_start) <= 0.0
        or float(cfg.sampling_power_mid) <= 0.0
        or float(cfg.sampling_power_end) <= 0.0
    ):
        raise ValueError("sampling power schedule endpoints must be positive.")
    if str(cfg.sampling_power_schedule) not in {"constant", "linear", "piecewise"}:
        raise ValueError("sampling_power_schedule must be 'constant', 'linear', or 'piecewise'.")
    if int(cfg.sampling_power_mid_iter) < 0 or int(cfg.sampling_power_end_iter) < 0:
        raise ValueError("sampling power schedule iteration thresholds must be non-negative.")
    if float(cfg.score_threshold_abs) < 0.0 or float(cfg.score_threshold_rel) < 0.0:
        raise ValueError("score thresholds must be non-negative.")
    if str(cfg.active_pool_rule) not in {"legacy_or", "max_threshold"}:
        raise ValueError("active_pool_rule must be 'legacy_or' or 'max_threshold'.")
    if str(cfg.active_pool_quantile_schedule) not in {"constant", "piecewise"}:
        raise ValueError("active_pool_quantile_schedule must be 'constant' or 'piecewise'.")
    if not (0.0 <= float(cfg.active_pool_min_quantile) <= float(cfg.active_pool_quantile) <= 1.0):
        raise ValueError("active pool quantiles must satisfy 0 <= min_quantile <= quantile <= 1.")
    for name, value in (
        ("active_pool_quantile_start", cfg.active_pool_quantile_start),
        ("active_pool_quantile_mid", cfg.active_pool_quantile_mid),
        ("active_pool_quantile_end", cfg.active_pool_quantile_end),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1.")
    if float(cfg.active_pool_rel_to_p95) < 0.0:
        raise ValueError("active_pool_rel_to_p95 must be non-negative.")
    if not (0.0 < float(cfg.active_pool_max_fraction_start) <= 1.0):
        raise ValueError("active_pool_max_fraction_start must be in (0, 1].")
    if not (0.0 < float(cfg.active_pool_max_fraction_end) <= 1.0):
        raise ValueError("active_pool_max_fraction_end must be in (0, 1].")
    if int(cfg.active_selection_min_iterations) < 0:
        raise ValueError("active_selection_min_iterations must be non-negative.")
    if str(cfg.w_ext_schedule) not in {"constant", "piecewise"}:
        raise ValueError("w_ext_schedule must be 'constant' or 'piecewise'.")
    if str(cfg.b_delta_gate_mode) not in {"none", "normal_sc_competition"}:
        raise ValueError("b_delta_gate_mode must be 'none' or 'normal_sc_competition'.")
    if str(cfg.q_boundary_gate_mode) not in {"psc", "uf_competition"}:
        raise ValueError("q_boundary_gate_mode must be 'psc' or 'uf_competition'.")
    if str(cfg.interior_filter_mode) not in {"off", "soft_penalty", "hard_exclude"}:
        raise ValueError("interior_filter_mode must be 'off', 'soft_penalty', or 'hard_exclude'.")
    if not (0.0 <= float(cfg.interior_penalty_early) <= 1.0):
        raise ValueError("interior_penalty_early must be between 0 and 1.")
    if not (0.0 <= float(cfg.interior_penalty_late) <= 1.0):
        raise ValueError("interior_penalty_late must be between 0 and 1.")

    if run_mode == "discovery":
        if domain_mode != "full":
            raise ValueError("discovery mode requires candidate_domain_mode='full'.")
        if cfg.finite_t_band_width is not None:
            raise ValueError("discovery mode must not enable finite_t_band_width or the finite-T prior band.")
    elif domain_mode == "prior_band" and cfg.finite_t_band_width is None:
        raise ValueError("refinement prior_band mode requires finite_t_band_width.")

    return cfg
