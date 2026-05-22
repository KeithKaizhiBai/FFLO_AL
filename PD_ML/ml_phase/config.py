from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ActiveLearningConfig:
    # Physical thresholds
    delta_eps: float = 1e-3
    q_eps: float = 1e-2
    eta_strong: float = 0.5
    boundary_margin: float = 0.08

    # Candidate space
    kt_min: float = 0.0
    kt_max: float = 0.56
    ja_min: float = 0.0
    ja_max: float = 2.12
    n_kt_candidates: int = 241
    n_ja_candidates: int = 321
    prioritize_kt_max: float = 0.5
    prioritize_ja_max: float = 1.2
    finite_t_band_width: float = 0.08

    # Acquisition
    w_cls_uncertainty: float = 1.0
    w_reg_uncertainty: float = 0.8
    w_delta_boundary: float = 1.0
    w_q_boundary: float = 1.0
    w_eta_boundary: float = 0.7
    w_gradient: float = 0.7
    w_diversity: float = 0.3
    diversity_min_dist: float = 0.015
    delta_scale: float = 0.04
    q_scale: float = 0.04
    eta_scale: float = 0.15

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
