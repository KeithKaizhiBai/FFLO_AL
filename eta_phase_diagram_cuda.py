from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from tfflo_1d_cuda import (
    StageTimer,
    bdg_hamiltonian_batch,
    complex_dtype_for,
    fermi_weighted_sum,
    maybe_set_linalg_backend,
    require_cuda,
    synchronize_if_cuda,
)

# ------ Configuration and Utilities ------
@dataclass
class EtaPhaseConfig:
    delta0: float = 1.0
    t: float = 1.0
    lambda_ry: float = 0.6
    lambda_rz: float = 0.6
    mu: float = 0.55
    u: float = 1.6
    omega_d: float = 10.0

    kt_min: float = 0.01
    kt_max: float = 0.5
    # Piecewise kT grid: dense on [kt_min, kt_dense_end], moderate on (kt_dense_end, kt_mid_end], coarser tail.
    kt_dense_end: float = 0.1
    kt_mid_end: float = 0.3
    n_kt_dense_01: int = 60
    n_kt_mid_03: int = 42
    n_kt_tail: int = 36

    ja_min: float = 0.01
    ja_max: float = 1.2
    # Piecewise JA grid: dense near phase transition (default JA ≈ 0.6).
    ja_refine_center: float = 0.6
    ja_refine_half_width: float = 0.15
    n_ja_left: int = 24
    n_ja_mid: int = 108
    n_ja_right: int = 24

    delta_min: float = 0.0
    delta_max: float = 0.6
    n_delta: int = 400

    q_min: float = -1.0
    q_max: float = 0.5
    n_q: int = 400

    n_k: int = 800

    kt_chunk: int = 8
    ja_chunk: int = 8
    delta_chunk: int = 4
    q_chunk: int = 100
    k_chunk: int = 200
    max_eig_batch: int = 10000

    # dtype: torch.dtype = torch.float32
    dtype: torch.dtype = torch.float64
    linalg_library: Literal["default", "cusolver", "magma"] = "cusolver"

    # Per-run outputs go under: data_phase_root / f"eta_phase_diagram_{run_tag}" /
    # Example: Data_Phase/eta_phase_diagram_nkt40_nja40_.../eta_timing_....json
    data_phase_root: str = "Data_Phase"
    save_npz: bool = True
    save_fig: bool = True

    def scaled(self) -> "EtaPhaseConfig":
        s = self.delta0
        return EtaPhaseConfig(
            delta0=s,
            t=self.t * s,
            lambda_ry=self.lambda_ry * s,
            lambda_rz=self.lambda_rz * s,
            mu=self.mu * s,
            u=self.u * s,
            omega_d=self.omega_d,
            kt_min=self.kt_min,
            kt_max=self.kt_max,
            kt_dense_end=self.kt_dense_end,
            kt_mid_end=self.kt_mid_end,
            n_kt_dense_01=self.n_kt_dense_01,
            n_kt_mid_03=self.n_kt_mid_03,
            n_kt_tail=self.n_kt_tail,
            ja_min=self.ja_min,
            ja_max=self.ja_max,
            ja_refine_center=self.ja_refine_center,
            ja_refine_half_width=self.ja_refine_half_width,
            n_ja_left=self.n_ja_left,
            n_ja_mid=self.n_ja_mid,
            n_ja_right=self.n_ja_right,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            n_delta=self.n_delta,
            q_min=self.q_min,
            q_max=self.q_max,
            n_q=self.n_q,
            n_k=self.n_k,
            kt_chunk=self.kt_chunk,
            ja_chunk=self.ja_chunk,
            delta_chunk=self.delta_chunk,
            q_chunk=self.q_chunk,
            k_chunk=self.k_chunk,
            max_eig_batch=self.max_eig_batch,
            dtype=self.dtype,
            linalg_library=self.linalg_library,
            data_phase_root=self.data_phase_root,
            save_npz=self.save_npz,
            save_fig=self.save_fig,
        )


def _dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.float32:
        return "float32"
    if dt == torch.float64:
        return "float64"
    return str(dt)


def _str_to_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s in ("float32", "fp32", "f32"):
        return torch.float32
    if s in ("float64", "fp64", "f64", "double"):
        return torch.float64
    raise argparse.ArgumentTypeError(f"Unsupported dtype: {s}")


def eta_config_to_jsonable(cfg: EtaPhaseConfig) -> Dict[str, Any]:
    d = {f.name: getattr(cfg, f.name) for f in fields(EtaPhaseConfig)}
    d["dtype"] = _dtype_to_str(cfg.dtype)
    return d


def eta_config_from_jsonable(d: Dict[str, Any]) -> EtaPhaseConfig:
    raw = dict(d)
    raw["dtype"] = _str_to_dtype(str(raw["dtype"]))
    return EtaPhaseConfig(**{k.name: raw[k.name] for k in fields(EtaPhaseConfig)})


def stable_config_hash(cfg: EtaPhaseConfig) -> str:
    payload = json.dumps(eta_config_to_jsonable(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write compressed npz atomically. Name must end in .npz or numpy appends another .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.partial{path.suffix}"
    if tmp.exists():
        tmp.unlink()
    np.savez_compressed(tmp, **arrays)
    if not tmp.is_file():
        raise RuntimeError(f"atomic_save_npz: temp file not created: {tmp}")
    if path.exists():
        path.unlink()
    tmp.replace(path)


# ----- Data Types and Computation -----
def make_eta_run_tag(cfg: EtaPhaseConfig, n_kt: int, n_ja: int, n_q: int) -> str:
    dt = "fp32" if cfg.dtype == torch.float32 else "fp64"
    cfg_hash = stable_config_hash(cfg)
    return (
        f"nkt{n_kt}_nja{n_ja}_nd{cfg.n_delta}_nq{n_q}_nk{cfg.n_k}"
        f"_kc{cfg.kt_chunk}_jc{cfg.ja_chunk}_dc{cfg.delta_chunk}"
        f"_qc{cfg.q_chunk}_kk{cfg.k_chunk}_eb{cfg.max_eig_batch}"
        f"_{dt}_lib{cfg.linalg_library}_cfg{cfg_hash}"
    )


def build_ja_vec(cfg: EtaPhaseConfig) -> np.ndarray:
    """JA grid with higher density near ja_refine_center (phase transition)."""
    c = cfg.ja_refine_center
    hw = cfg.ja_refine_half_width
    lo = max(cfg.ja_min, c - hw)
    hi = min(cfg.ja_max, c + hw)
    parts: List[np.ndarray] = []
    if cfg.ja_min < lo - 1e-12:
        parts.append(np.linspace(cfg.ja_min, lo, cfg.n_ja_left, endpoint=False))
    if hi > lo + 1e-12:
        parts.append(np.linspace(lo, hi, cfg.n_ja_mid))
    if cfg.ja_max > hi + 1e-12:
        parts.append(np.linspace(hi, cfg.ja_max, cfg.n_ja_right + 1)[1:])
    if not parts:
        ntot = max(3, cfg.n_ja_left + cfg.n_ja_mid + cfg.n_ja_right)
        return np.linspace(cfg.ja_min, cfg.ja_max, ntot, dtype=np.float64)
    ja = np.unique(np.concatenate(parts))
    return np.sort(ja).astype(np.float64)


def build_kt_vec(cfg: EtaPhaseConfig) -> np.ndarray:
    """kT grid: dense on [kt_min, kt_dense_end], moderate on (kt_dense_end, kt_mid_end], coarser beyond."""
    d1, d2 = cfg.kt_dense_end, cfg.kt_mid_end
    km, kM = cfg.kt_min, cfg.kt_max
    parts: List[np.ndarray] = []

    upper_a = min(d1, kM)
    if km < upper_a - 1e-12:
        single = kM <= d1 + 1e-12
        parts.append(np.linspace(km, upper_a, cfg.n_kt_dense_01, endpoint=single))

    if kM > d1 + 1e-12:
        lo_b = max(d1, km)
        hi_b = min(d2, kM)
        if hi_b > lo_b + 1e-12:
            single = kM <= d2 + 1e-12
            parts.append(np.linspace(lo_b, hi_b, cfg.n_kt_mid_03, endpoint=single))

    if kM > d2 + 1e-12:
        lo_c = max(d2, km)
        if kM > lo_c + 1e-12:
            parts.append(np.linspace(lo_c, kM, cfg.n_kt_tail))

    if not parts:
        return np.linspace(km, kM, max(2, cfg.n_kt_dense_01), dtype=np.float64)
    kt = np.unique(np.concatenate(parts))
    return np.sort(kt).astype(np.float64)


def build_q_vec(cfg: EtaPhaseConfig) -> np.ndarray:
    """Uniform q grid on [q_min, q_max]."""
    return np.linspace(cfg.q_min, cfg.q_max, cfg.n_q, dtype=np.float64)


def eigvalsh_limited(h: torch.Tensor, max_batch: int) -> torch.Tensor:
    """Run eigvalsh in smaller flattened batches to cap cuSOLVER workspace."""
    if max_batch <= 0:
        return torch.linalg.eigvalsh(h).real
    matrix_shape = h.shape[-2:]
    if matrix_shape != (4, 4):
        raise ValueError(f"Expected (..., 4, 4) matrices, got {matrix_shape}")
    batch_shape = h.shape[:-2]
    n_mats = math.prod(batch_shape)
    flat = h.reshape(n_mats, 4, 4)
    evals = torch.empty((n_mats, 4), dtype=h.real.dtype, device=h.device)
    for start in range(0, n_mats, max_batch):
        end = min(start + max_batch, n_mats)
        evals[start:end] = torch.linalg.eigvalsh(flat[start:end]).real
    return evals.reshape(*batch_shape, 4)


def iter_kt_ja_tiles(n_kt: int, n_ja: int, kt_chunk: int, ja_chunk: int) -> List[Tuple[int, int, int, int]]:
    """All (kt_start, kt_end, ja_start, ja_end) tiles in deterministic order."""
    tiles: List[Tuple[int, int, int, int]] = []
    for kt_start in range(0, n_kt, kt_chunk):
        kt_end = min(kt_start + kt_chunk, n_kt)
        for ja_start in range(0, n_ja, ja_chunk):
            ja_end = min(ja_start + ja_chunk, n_ja)
            tiles.append((kt_start, kt_end, ja_start, ja_end))
    return tiles


def shard_npz_path(output_dir: Path, rank: int, world_size: int, run_tag: str) -> Path:
    return output_dir / f"eta_shard_rank{rank:03d}_of{world_size:03d}_{run_tag}.npz"


def shard_timing_path(output_dir: Path, rank: int, world_size: int, run_tag: str) -> Path:
    return output_dir / f"eta_timing_rank{rank:03d}_of{world_size:03d}_{run_tag}.json"


def _finite_T_phase_boundary_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Values from finite_T_phase_diagram.m (same numeric tables as MATLAB)."""
    T_fflo_ns = np.array(
        [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56],
        dtype=np.float64,
    )
    JA_fflo_ns = np.array(
        [2.12, 1.733, 1.5, 1.32, 1.16, 0.9, 0.78, 0.733, 0.7, 0.667, 0.59, 0.4, 0.178, 0.0],
        dtype=np.float64,
    )
    T1st = np.array([0.01, 0.04, 0.05], dtype=np.float64)
    JA1st = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    T2nd = np.array([0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float64)
    JA2nd = np.array([0.6, 0.6, 0.62, 0.6277, 0.63, 0.628, 0.617, 0.598, 0.565], dtype=np.float64)
    return T_fflo_ns, JA_fflo_ns, T1st, JA1st, T2nd, JA2nd


def plot_finite_T_phase_boundaries(ax: plt.Axes, cfg: EtaPhaseConfig) -> None:
    """Overlay three classified boundaries from finite_T_phase_diagram.m."""
    T_fflo_ns, JA_fflo_ns, T1st, JA1st, T2nd, JA2nd = _finite_T_phase_boundary_arrays()
    s0 = float(cfg.delta0)
    # Match the heatmap axes in plot_eta_phase_diagram: x=T/delta0, y=JA/delta0.
    ax.plot(
        T_fflo_ns / s0,
        JA_fflo_ns / s0,
        "ks-",
        ms=4,
        lw=1.5,
        label="tFFLO–normal boundary",
    )
    ax.plot(
        T1st / s0,
        JA1st / s0,
        color="red",
        linestyle=":",
        marker="D",
        ms=4,
        lw=1.5,
        label="cFFLO–tFFLO (1st order)",
    )
    ax.plot(
        T2nd / s0,
        JA2nd / s0,
        color="green",
        linestyle="-.",
        marker="o",
        ms=4,
        lw=1.5,
        label="cFFLO–tFFLO (2nd order)",
    )


# ----- Function for plotting -----
def bwr_cmap_adaptive(cmin: float, cmax: float, m: int = 256) -> np.ndarray:
    if cmin >= 0:
        cmap = np.stack([np.ones(m), np.linspace(1, 0, m), np.linspace(1, 0, m)], axis=1)
    elif cmax <= 0:
        cmap = np.stack([np.linspace(0, 1, m), np.linspace(0, 1, m), np.ones(m)], axis=1)
    else:
        zero_ratio = abs(cmin) / (cmax - cmin)
        n_blue = max(1, round(m * zero_ratio))
        n_red = max(1, m - n_blue)
        b = np.stack([np.linspace(0, 1, n_blue), np.linspace(0, 1, n_blue), np.ones(n_blue)], axis=1)
        r = np.stack([np.ones(n_red), np.linspace(1 - 1 / n_red, 0, n_red), np.linspace(1 - 1 / n_red, 0, n_red)], axis=1)
        cmap = np.concatenate([b, r], axis=0)
    return np.clip(cmap, 0.0, 1.0)


def cell_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Bin edges for pcolormesh such that cell i is centered at centers[i] (handles non-uniform spacing)."""
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size == 0:
        raise ValueError("cell_edges_from_centers: empty centers")
    if c.size == 1:
        span = 0.01 * max(abs(float(c[0])), 1.0)
        return np.array([c[0] - span / 2, c[0] + span / 2], dtype=np.float64)
    inner = 0.5 * (c[:-1] + c[1:])
    left = float(c[0] - (inner[0] - c[0]))
    right = float(c[-1] + (c[-1] - inner[-1]))
    return np.concatenate([[left], inner, [right]])

# ------ Main Computation Functions ------
def compute_omega_min_q_batch(
    kt_batch: torch.Tensor,
    ja_batch: torch.Tensor,
    cfg: EtaPhaseConfig,
    k_vec: torch.Tensor,
    q_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = k_vec.device
    ctype = complex_dtype_for(cfg.dtype)

    delta_vec = torch.linspace(cfg.delta_min, cfg.delta_max, cfg.n_delta, dtype=cfg.dtype, device=device)
    n_q = int(q_vec.numel())

    mask = torch.abs(cfg.t * torch.cos(k_vec) - cfg.mu) < cfg.omega_d
    valid_k = k_vec[mask]
    if valid_k.numel() == 0:
        raise RuntimeError("No valid k-point satisfies Debye cutoff.")

    nkt, nja = kt_batch.numel(), ja_batch.numel()
    omega_min_q = torch.empty((nkt, nja, n_q), dtype=cfg.dtype, device=device)
    delta_opt_q = torch.empty((nkt, nja, n_q), dtype=cfg.dtype, device=device)
    nk = float(cfg.n_k)

    kt_b = torch.clamp(kt_batch.view(nkt, 1, 1, 1, 1, 1), min=1e-6)
    ja_b = ja_batch.view(1, nja, 1, 1, 1)

    for q_start in range(0, n_q, cfg.q_chunk):
        q_end = min(q_start + cfg.q_chunk, n_q)
        q_part = q_vec[q_start:q_end]
        q_b = q_part.view(1, 1, 1, q_part.numel(), 1)

        omega_best = torch.full((nkt, nja, q_part.numel()), torch.inf, dtype=cfg.dtype, device=device)
        delta_best = torch.zeros((nkt, nja, q_part.numel()), dtype=cfg.dtype, device=device)

        for delta_start in range(0, cfg.n_delta, cfg.delta_chunk):
            delta_end = min(delta_start + cfg.delta_chunk, cfg.n_delta)
            delta_part = delta_vec[delta_start:delta_end]
            d_b = delta_part.view(1, 1, delta_part.numel(), 1, 1)
            delta_term = (delta_part**2 / cfg.u).view(1, 1, delta_part.numel(), 1)

            int_sum = torch.zeros((nkt, nja, delta_part.numel(), q_part.numel()), dtype=cfg.dtype, device=device)
            int_sum0 = torch.zeros_like(int_sum)

            for k_start in range(0, valid_k.numel(), cfg.k_chunk):
                k_end = min(k_start + cfg.k_chunk, valid_k.numel())
                k_part = valid_k[k_start:k_end]
                k_b = k_part.view(1, 1, 1, 1, k_part.numel())

                h = bdg_hamiltonian_batch(
                    k=k_b,
                    q=q_b,
                    delta=d_b,
                    t=cfg.t,
                    lambda_ry=cfg.lambda_ry,
                    lambda_rz=cfg.lambda_rz,
                    ja=ja_b,
                    mu=cfg.mu,
                    complex_dtype=ctype,
                )
                e = eigvalsh_limited(h, cfg.max_eig_batch)
                int_sum += torch.sum(fermi_weighted_sum(e, kt_b), dim=-1)
                del h, e

                h0 = bdg_hamiltonian_batch(
                    k=k_b,
                    q=q_b,
                    delta=torch.zeros_like(d_b),
                    t=cfg.t,
                    lambda_ry=cfg.lambda_ry,
                    lambda_rz=cfg.lambda_rz,
                    ja=ja_b,
                    mu=cfg.mu,
                    complex_dtype=ctype,
                )
                e0 = eigvalsh_limited(h0, cfg.max_eig_batch)
                int_sum0 += torch.sum(fermi_weighted_sum(e0, kt_b), dim=-1)
                del h0, e0

            omega = 0.5 * int_sum / nk + delta_term - 0.5 * int_sum0 / nk
            omega_delta_min, idx_min = torch.min(omega, dim=2)
            better = omega_delta_min < omega_best
            omega_best = torch.where(better, omega_delta_min, omega_best)
            delta_candidate = delta_part[idx_min]
            delta_best = torch.where(better, delta_candidate, delta_best)
            del int_sum, int_sum0, omega, omega_delta_min, idx_min, better, delta_candidate

        omega_min_q[:, :, q_start:q_end] = omega_best
        delta_opt_q[:, :, q_start:q_end] = delta_best

    omega_global, idx_q = torch.min(omega_min_q, dim=2)
    q_opt = q_vec[idx_q]
    delta_opt = torch.gather(delta_opt_q, dim=2, index=idx_q.unsqueeze(-1)).squeeze(-1)
    _ = omega_global
    return omega_min_q, delta_opt_q, q_opt, delta_opt

# Compute the current j(q) from the omega(q) data using finite difference
def compute_current_from_omega(omega_min_q: np.ndarray, q_vec: np.ndarray) -> np.ndarray:
    nq = q_vec.shape[0]
    j_q = np.zeros_like(omega_min_q)
    if nq < 2:
        return j_q
    if nq == 2:
        dq0 = float(q_vec[1] - q_vec[0])
        slope = (omega_min_q[..., 1] - omega_min_q[..., 0]) / dq0
        j_q[..., 0] = slope
        j_q[..., 1] = slope
        return j_q
    j_q[..., 1:-1] = (omega_min_q[..., 2:] - omega_min_q[..., :-2]) / (q_vec[2:] - q_vec[:-2])
    j_q[..., 0] = (omega_min_q[..., 1] - omega_min_q[..., 0]) / (q_vec[1] - q_vec[0])
    j_q[..., -1] = (omega_min_q[..., -1] - omega_min_q[..., -2]) / (q_vec[-1] - q_vec[-2])
    return j_q

# Analyze j(q) to find the critical currents in positive and negative directions, and compute eta
def find_eta_from_jq(j_q: np.ndarray, q_vec: np.ndarray, idx_q_opt: int) -> Tuple[float, float, float]:
    n_q = j_q.shape[0]
    i_c_plus = 0.0
    i_c_minus = 0.0
    q_plus = np.nan
    q_minus = np.nan

    for iq in range(idx_q_opt, n_q - 1):
        if j_q[iq] > 0 and j_q[iq + 1] <= j_q[iq]:
            i_c_plus = float(j_q[iq])
            q_plus = float(q_vec[iq])
            break
    if np.isnan(q_plus) and j_q[-1] > 0 and j_q[-1] > j_q[-2]:
        i_c_plus = float(j_q[-1])
        q_plus = float(q_vec[-1])

    for iq in range(idx_q_opt, 0, -1):
        if j_q[iq] < 0 and j_q[iq - 1] >= j_q[iq]:
            i_c_minus = float(j_q[iq])
            q_minus = float(q_vec[iq])
            break
    if np.isnan(q_minus) and j_q[0] < 0 and j_q[0] < j_q[1]:
        i_c_minus = float(j_q[0])
        q_minus = float(q_vec[0])

    if i_c_plus == 0 and i_c_minus == 0:
        eta = 0.0
    elif i_c_plus == 0:
        eta = -1.0
    elif i_c_minus == 0:
        eta = 1.0
    else:
        eta = (abs(i_c_plus) - abs(i_c_minus)) / (abs(i_c_plus) + abs(i_c_minus))

    _ = (q_plus, q_minus)
    return eta, i_c_plus, i_c_minus

# ----- Main Function to Run the Phase Diagram Computation -----
def plot_eta_phase_diagram(
    cfg: EtaPhaseConfig,
    kt_vec: np.ndarray,
    ja_vec: np.ndarray,
    eta_matrix: np.ndarray,
    run_tag: str,
    output_dir: Path,
) -> None:
    """Plot eta(J_A, kT) on non-uniform grids using pcolormesh (imshow assumes uniform index spacing)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    min_eta = float(np.min(eta_matrix))
    max_eta = float(np.max(eta_matrix))
    if min_eta == 0 and max_eta == 0:
        min_eta, max_eta = -1e-6, 1e-6

    cmap = plt.matplotlib.colors.ListedColormap(bwr_cmap_adaptive(min_eta, max_eta, 256))
    kt_plot = np.asarray(kt_vec, dtype=np.float64) / cfg.delta0
    ja_plot = np.asarray(ja_vec, dtype=np.float64) / cfg.delta0
    kt_e = cell_edges_from_centers(kt_plot)
    ja_e = cell_edges_from_centers(ja_plot)

    mesh = ax.pcolormesh(
        kt_e,
        ja_e,
        eta_matrix,
        cmap=cmap,
        vmin=min_eta,
        vmax=max_eta,
        shading="flat",
    )
    fig.colorbar(mesh, ax=ax)
    ax.set_xlabel("k_B T / t")
    ax.set_ylabel("J_A / t")
    ax.set_title("Superconducting Diode Effect Efficiency eta Phase Diagram")

    plot_finite_T_phase_boundaries(ax, cfg)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_aspect("auto")
    fig.tight_layout()
    fig.savefig(output_dir / f"eta_phase_diagram_{run_tag}.png", dpi=240)
    fig.savefig(output_dir / f"eta_phase_diagram_{run_tag}.pdf")
    plt.close(fig)


def _expected_meta(
    run_tag: str,
    rank: int,
    world_size: int,
    cfg: EtaPhaseConfig,
    n_kt: int,
    n_ja: int,
    n_q: int,
) -> Dict[str, Any]:
    return {
        "run_tag": run_tag,
        "rank": rank,
        "world_size": world_size,
        "n_kt": n_kt,
        "n_ja": n_ja,
        "n_q": n_q,
        "config": eta_config_to_jsonable(cfg),
    }


def _meta_dicts_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if k == "config":
            if a["config"] != b["config"]:
                return False
        elif a[k] != b[k]:
            return False
    return True


def _vectors_match_shard(
    kt_vec: np.ndarray,
    ja_vec: np.ndarray,
    q_vec: np.ndarray,
    shard: Any,
) -> bool:
    try:
        return (
            np.array_equal(np.asarray(shard["kT_vec"]), np.asarray(kt_vec))
            and np.array_equal(np.asarray(shard["JA_vec"]), np.asarray(ja_vec))
            and np.array_equal(np.asarray(shard["q_vec"]), np.asarray(q_vec))
        )
    except KeyError:
        return False


def expected_computed_mask(
    n_kt: int,
    n_ja: int,
    kt_chunk: int,
    ja_chunk: int,
    rank: int,
    world_size: int,
) -> np.ndarray:
    mask = np.zeros((n_ja, n_kt), dtype=np.bool_)
    tiles = iter_kt_ja_tiles(n_kt, n_ja, kt_chunk, ja_chunk)
    for tile_idx, (kt_start, kt_end, ja_start, ja_end) in enumerate(tiles):
        if tile_idx % world_size == rank:
            mask[ja_start:ja_end, kt_start:kt_end] = True
    return mask


def validate_eta_shard(
    path: Path,
    *,
    expected_meta: Dict[str, Any],
    kt_vec: np.ndarray,
    ja_vec: np.ndarray,
    q_vec: np.ndarray,
    expected_mask: np.ndarray,
) -> Tuple[bool, str]:
    required_keys = {
        "meta_json_utf8",
        "kT_vec",
        "JA_vec",
        "q_vec",
        "eta_matrix",
        "q_opt_matrix",
        "delta_opt_matrix",
        "ic_plus_matrix",
        "ic_minus_matrix",
        "computed_mask",
    }
    if not path.is_file():
        return False, f"missing shard: {path}"
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = required_keys.difference(z.files)
            if missing:
                return False, f"missing keys: {sorted(missing)}"
            meta = json.loads(bytes(z["meta_json_utf8"]).decode("utf-8"))
            if not _meta_dicts_equal(meta, expected_meta):
                return False, "metadata mismatch"
            if not _vectors_match_shard(kt_vec, ja_vec, q_vec, z):
                return False, "grid vectors mismatch"

            expected_shape = expected_mask.shape
            cm = np.asarray(z["computed_mask"], dtype=np.bool_)
            if cm.shape != expected_shape:
                return False, f"computed_mask shape {cm.shape} != {expected_shape}"
            if not np.array_equal(cm, expected_mask):
                return False, "computed_mask does not match this rank's expected tiles"

            for key in ("eta_matrix", "q_opt_matrix", "delta_opt_matrix", "ic_plus_matrix", "ic_minus_matrix"):
                vals = np.asarray(z[key])
                if vals.shape != expected_shape:
                    return False, f"{key} shape {vals.shape} != {expected_shape}"
                if not np.all(np.isfinite(vals[cm])):
                    return False, f"{key} has non-finite values on computed cells"
    except Exception as exc:
        return False, f"failed to read shard: {exc}"
    return True, "ok"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def run_eta_phase_diagram(
    cfg: EtaPhaseConfig,
    *,
    rank: int = 0,
    world_size: int = 1,
    resume: bool = False,
    run_tag_override: str | None = None,
    shard_dir: Path | None = None,
    profile: bool = False,
) -> Dict[str, np.ndarray] | None:
    """
    Compute eta phase diagram. If world_size > 1, writes per-rank shard under output_dir
    (or shard_dir); merge with merge_eta_phase_diagram_shards. world_size == 1 writes final
    npz and figure directly (legacy behavior).
    """
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank={rank} for world_size={world_size}")

    cfg = cfg.scaled()
    kt_vec = build_kt_vec(cfg)
    ja_vec = build_ja_vec(cfg)
    q_vec = build_q_vec(cfg)
    n_kt, n_ja, n_q = int(kt_vec.shape[0]), int(ja_vec.shape[0]), int(q_vec.shape[0])
    run_tag = run_tag_override or make_eta_run_tag(cfg, n_kt, n_ja, n_q)
    output_dir = Path(cfg.data_phase_root) / f"eta_phase_diagram_{run_tag}"
    out_for_shards = shard_dir if shard_dir is not None else output_dir
    out_for_shards.mkdir(parents=True, exist_ok=True)

    shard_path = shard_npz_path(out_for_shards, rank, world_size, run_tag)
    timing_path = shard_timing_path(out_for_shards, rank, world_size, run_tag)

    if world_size > 1 and resume and shard_path.is_file():
        expected = _expected_meta(run_tag, rank, world_size, cfg, n_kt, n_ja, n_q)
        expected_mask = expected_computed_mask(n_kt, n_ja, cfg.kt_chunk, cfg.ja_chunk, rank, world_size)
        is_valid, reason = validate_eta_shard(
            shard_path,
            expected_meta=expected,
            kt_vec=kt_vec,
            ja_vec=ja_vec,
            q_vec=q_vec,
            expected_mask=expected_mask,
        )
        if is_valid:
            print(f"[eta] resume: skipping rank {rank}, shard OK: {shard_path}")
            return None
        print(f"[eta] resume: recomputing rank {rank}; invalid shard ({reason})")

    print(f"Saving eta phase outputs under: {output_dir.resolve()}")
    if world_size > 1:
        print(f"Shard dir (rank {rank}): {out_for_shards.resolve()}")
    print(
        f"Grid: n_kT={n_kt} (kT: denser [{cfg.kt_min:g},{cfg.kt_dense_end}], moderate "
        f"({cfg.kt_dense_end},{cfg.kt_mid_end}], coarser tail); "
        f"n_JA={n_ja} near JA={cfg.ja_refine_center}; n_q={n_q} uniform) | "
        f"rank {rank}/{world_size}"
    )

    prof: Any = None
    if profile:
        import cProfile

        prof = cProfile.Profile()
        prof.enable()

    device = require_cuda()
    maybe_set_linalg_backend(cfg)

    torch.cuda.empty_cache()

    kt_vec_f = kt_vec.astype(np.float32, copy=False)
    k_vec = torch.linspace(-math.pi, math.pi, cfg.n_k, dtype=cfg.dtype, device=device)
    q_vec_t = torch.as_tensor(q_vec, device=device, dtype=cfg.dtype)

    tiles = iter_kt_ja_tiles(n_kt, n_ja, cfg.kt_chunk, cfg.ja_chunk)
    my_tiles = [t for i, t in enumerate(tiles) if i % world_size == rank]
    my_pts = sum((te - ts) * (je - js) for ts, te, js, je in my_tiles)
    total_pts = n_kt * n_ja

    nan_f = np.float32(np.nan)
    eta_matrix = np.full((n_ja, n_kt), nan_f, dtype=np.float32)
    q_opt_matrix = np.full_like(eta_matrix, nan_f)
    delta_opt_matrix = np.full_like(eta_matrix, nan_f)
    i_c_plus_matrix = np.full_like(eta_matrix, nan_f)
    i_c_minus_matrix = np.full_like(eta_matrix, nan_f)
    computed_mask = np.zeros((n_ja, n_kt), dtype=np.bool_)

    timer = StageTimer()
    overall_t0 = time.perf_counter()

    done_pts = 0
    next_pct = 10
    loop_t0 = time.perf_counter()

    for tile_idx, (kt_start, kt_end, ja_start, ja_end) in enumerate(my_tiles):
        kt_part_np = kt_vec_f[kt_start:kt_end]
        kt_part = torch.as_tensor(kt_part_np, device=device, dtype=cfg.dtype)
        ja_part_np = ja_vec[ja_start:ja_end]
        ja_part = torch.as_tensor(ja_part_np, device=device, dtype=cfg.dtype)

        t0 = time.perf_counter()
        omega_min_q_t, _delta_opt_q_t, q_opt_t, delta_opt_t = compute_omega_min_q_batch(
            kt_part, ja_part, cfg, k_vec, q_vec_t
        )
        synchronize_if_cuda(device)
        timer.add("gpu_omega_scan", time.perf_counter() - t0)

        t1 = time.perf_counter()
        omega_min_q = omega_min_q_t.to("cpu").numpy()
        q_opt = q_opt_t.to("cpu").numpy()
        delta_opt = delta_opt_t.to("cpu").numpy()
        j_q = compute_current_from_omega(omega_min_q, q_vec)

        for ikt in range(kt_end - kt_start):
            for ija in range(ja_end - ja_start):
                iq_opt = int(np.argmin(np.abs(q_vec - q_opt[ikt, ija])))
                eta, i_plus, i_minus = find_eta_from_jq(j_q[ikt, ija], q_vec, iq_opt)

                row = ja_start + ija
                col = kt_start + ikt
                eta_matrix[row, col] = eta
                q_opt_matrix[row, col] = q_opt[ikt, ija]
                delta_opt_matrix[row, col] = delta_opt[ikt, ija]
                i_c_plus_matrix[row, col] = i_plus
                i_c_minus_matrix[row, col] = i_minus
                computed_mask[row, col] = True
        timer.add("eta_postprocess", time.perf_counter() - t1)

        done_pts += (kt_end - kt_start) * (ja_end - ja_start)
        if my_pts > 0:
            pct = int(done_pts * 100 / my_pts)
            while pct >= next_pct and next_pct <= 100:
                elapsed = time.perf_counter() - loop_t0
                eta_s = elapsed * (my_pts - done_pts) / max(done_pts, 1)
                print(
                    f"[eta r{rank}] {next_pct:3d}% | done {done_pts}/{my_pts} | "
                    f"tile {tile_idx + 1}/{len(my_tiles)} kT[{kt_start}:{kt_end}) JA[{ja_start}:{ja_end}) | "
                    f"elapsed {elapsed:.1f}s | ETA {eta_s:.1f}s"
                )
                next_pct += 10

    if profile and prof is not None:
        prof.disable()
        import pstats
        from io import StringIO

        s = StringIO()
        pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(40)
        stats_path = out_for_shards / f"eta_profile_rank{rank:03d}_{run_tag}.txt"
        stats_path.write_text(s.getvalue(), encoding="utf-8")
        print(f"[eta] cProfile stats written to {stats_path}")

    total_s = time.perf_counter() - overall_t0
    print("\n=== Timing (seconds) ===")
    print(f"total: {total_s:.3f}")
    for line in timer.summary_lines():
        print(line)

    meta = _expected_meta(run_tag, rank, world_size, cfg, n_kt, n_ja, n_q)
    meta_json = json.dumps(meta, ensure_ascii=False)
    meta_json_utf8 = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

    timing_rank = {
        "run_tag": run_tag,
        "rank": rank,
        "world_size": world_size,
        "output_dir": str(output_dir.resolve()),
        "shard_dir": str(out_for_shards.resolve()),
        "data_phase_root": cfg.data_phase_root,
        "total_s": total_s,
        "stage_totals_s": timer.stage_totals_s,
        "stage_counts": timer.stage_counts,
        "config": meta["config"],
    }

    if world_size > 1:
        atomic_save_npz(
            shard_path,
            meta_json_utf8=meta_json_utf8,
            kT_vec=kt_vec.astype(np.float64),
            JA_vec=ja_vec.astype(np.float64),
            q_vec=q_vec.astype(np.float64),
            eta_matrix=eta_matrix,
            q_opt_matrix=q_opt_matrix,
            delta_opt_matrix=delta_opt_matrix,
            ic_plus_matrix=i_c_plus_matrix,
            ic_minus_matrix=i_c_minus_matrix,
            computed_mask=computed_mask,
        )
        _write_json_atomic(timing_path, timing_rank)
        print(f"[eta] wrote shard {shard_path}")
        return None

    # world_size == 1: final outputs on this process
    t_plot = time.perf_counter()
    if cfg.save_fig:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_eta_phase_diagram(cfg, kt_vec, ja_vec, eta_matrix, run_tag, output_dir)
    timer.add("plot_phase_diagram", time.perf_counter() - t_plot)

    results = {
        "kT_vec": kt_vec.astype(np.float32, copy=False),
        "JA_vec": ja_vec,
        "q_vec": q_vec,
        "eta_matrix": eta_matrix,
        "q_opt_matrix": q_opt_matrix,
        "delta_opt_matrix": delta_opt_matrix,
        "ic_plus_matrix": i_c_plus_matrix,
        "ic_minus_matrix": i_c_minus_matrix,
        "delta0": np.array(cfg.delta0, dtype=np.float64),
    }

    if cfg.save_npz:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(output_dir / f"eta_phase_diagram_{run_tag}.npz", **results)

    timing = {
        "run_tag": run_tag,
        "rank": 0,
        "world_size": 1,
        "output_dir": str(output_dir.resolve()),
        "data_phase_root": cfg.data_phase_root,
        "total_s": total_s,
        "stage_totals_s": timer.stage_totals_s,
        "stage_counts": timer.stage_counts,
        "config": {
            "n_kt": n_kt,
            "n_ja": n_ja,
            "n_q": n_q,
            "n_delta": cfg.n_delta,
            "n_k": cfg.n_k,
            "kt_dense_end": cfg.kt_dense_end,
            "kt_mid_end": cfg.kt_mid_end,
            "n_kt_dense_01": cfg.n_kt_dense_01,
            "n_kt_mid_03": cfg.n_kt_mid_03,
            "n_kt_tail": cfg.n_kt_tail,
            "ja_refine_center": cfg.ja_refine_center,
            "ja_refine_half_width": cfg.ja_refine_half_width,
            "n_ja_left": cfg.n_ja_left,
            "n_ja_mid": cfg.n_ja_mid,
            "n_ja_right": cfg.n_ja_right,
            "q_min": cfg.q_min,
            "q_max": cfg.q_max,
            "kt_chunk": cfg.kt_chunk,
            "ja_chunk": cfg.ja_chunk,
            "delta_chunk": cfg.delta_chunk,
            "q_chunk": cfg.q_chunk,
            "k_chunk": cfg.k_chunk,
            "max_eig_batch": cfg.max_eig_batch,
            "dtype": str(cfg.dtype),
            "linalg_library": cfg.linalg_library,
        },
    }
    with (output_dir / f"eta_timing_{run_tag}.json").open("w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    return results


def merge_eta_phase_diagram_shards(
    cfg: EtaPhaseConfig,
    run_tag: str,
    *,
    world_size: int,
    shard_dir: Path | None = None,
) -> Dict[str, np.ndarray]:
    """Load all rank shards, validate, merge, write final npz/timing and optional figure."""
    cfg = cfg.scaled()
    output_dir = Path(cfg.data_phase_root) / f"eta_phase_diagram_{run_tag}"
    out_for_shards = shard_dir if shard_dir is not None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shards: List[Any] = []
    paths: List[Path] = []
    metas: List[Dict[str, Any]] = []

    for rank in range(world_size):
        p = shard_npz_path(out_for_shards, rank, world_size, run_tag)
        if not p.is_file():
            raise FileNotFoundError(f"Missing shard for merge: {p}")
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta_json_utf8"]).decode("utf-8"))
            if meta.get("run_tag") != run_tag or meta.get("world_size") != world_size or meta.get("rank") != rank:
                raise ValueError(f"Shard metadata mismatch in {p}: {meta}")
            cfg_shard = eta_config_from_jsonable(meta["config"])
            if eta_config_to_jsonable(cfg_shard) != eta_config_to_jsonable(cfg):
                raise ValueError(f"Shard config does not match current cfg for merge (rank {rank})")
            paths.append(p)
            # Copy arrays out before closing the npz (context manager closes file)
            shards.append({k: np.array(z[k]) for k in z.files})
            metas.append(meta)

    n_kt = int(metas[0]["n_kt"])
    n_ja = int(metas[0]["n_ja"])
    n_q = int(metas[0]["n_q"])
    for m in metas[1:]:
        if int(m["n_kt"]) != n_kt or int(m["n_ja"]) != n_ja or int(m["n_q"]) != n_q:
            raise ValueError("Inconsistent n_kt / n_ja / n_q across shards")

    kt_vec = np.asarray(shards[0]["kT_vec"])
    ja_vec = np.asarray(shards[0]["JA_vec"])
    q_vec = np.asarray(shards[0]["q_vec"])
    for z in shards[1:]:
        if not (
            np.array_equal(np.asarray(z["kT_vec"]), kt_vec)
            and np.array_equal(np.asarray(z["JA_vec"]), ja_vec)
            and np.array_equal(np.asarray(z["q_vec"]), q_vec)
        ):
            raise ValueError("kT_vec / JA_vec / q_vec differ across shards")

    mask_sum = np.zeros((n_ja, n_kt), dtype=np.int32)
    merged_eta = np.full((n_ja, n_kt), np.nan, dtype=np.float64)
    merged_qo = np.full_like(merged_eta, np.nan)
    merged_d = np.full_like(merged_eta, np.nan)
    merged_ip = np.full_like(merged_eta, np.nan)
    merged_im = np.full_like(merged_eta, np.nan)

    for z in shards:
        cm = np.asarray(z["computed_mask"], dtype=np.bool_)
        if cm.shape != (n_ja, n_kt):
            raise ValueError(f"computed_mask shape {cm.shape} != ({n_ja}, {n_kt})")
        mask_sum += cm.astype(np.int32)
        for tgt, key in (
            (merged_eta, "eta_matrix"),
            (merged_qo, "q_opt_matrix"),
            (merged_d, "delta_opt_matrix"),
            (merged_ip, "ic_plus_matrix"),
            (merged_im, "ic_minus_matrix"),
        ):
            vals = np.asarray(z[key], dtype=np.float64)
            tgt[cm] = vals[cm]

    if not np.all(mask_sum == 1):
        bad = np.argwhere(mask_sum != 1)
        raise RuntimeError(
            f"Merge failed: expected each cell covered exactly once, got {bad.shape[0]} bad cells "
            f"(mask_sum min/max = {mask_sum.min()}/{mask_sum.max()})"
        )

    eta_f32 = merged_eta.astype(np.float32)
    qo_f32 = merged_qo.astype(np.float32)
    d_f32 = merged_d.astype(np.float32)
    ip_f32 = merged_ip.astype(np.float32)
    im_f32 = merged_im.astype(np.float32)

    results = {
        "kT_vec": kt_vec.astype(np.float32, copy=False),
        "JA_vec": ja_vec,
        "q_vec": q_vec,
        "eta_matrix": eta_f32,
        "q_opt_matrix": qo_f32,
        "delta_opt_matrix": d_f32,
        "ic_plus_matrix": ip_f32,
        "ic_minus_matrix": im_f32,
        "delta0": np.array(cfg.delta0, dtype=np.float64),
    }

    if cfg.save_npz:
        np.savez(output_dir / f"eta_phase_diagram_{run_tag}.npz", **results)

    t_plot = time.perf_counter()
    if cfg.save_fig:
        plot_eta_phase_diagram(cfg, kt_vec, ja_vec, eta_f32, run_tag, output_dir)
    plot_dt = time.perf_counter() - t_plot

    stage_totals: Dict[str, float] = {"merge_io": 0.0, "plot_phase_diagram": float(plot_dt)}
    stage_counts: Dict[str, int] = {"merge_io": len(shards), "plot_phase_diagram": 1 if cfg.save_fig else 0}
    timing_paths = [shard_timing_path(out_for_shards, r, world_size, run_tag) for r in range(world_size)]
    per_rank_totals: List[float] = []
    for tp in timing_paths:
        if tp.is_file():
            per_rank_totals.append(float(_load_json(tp).get("total_s", 0.0)))
    timing_merged = {
        "run_tag": run_tag,
        "world_size": world_size,
        "output_dir": str(output_dir.resolve()),
        "shard_dir": str(out_for_shards.resolve()),
        "shard_files": [str(p.resolve()) for p in paths],
        "merge_stage_totals_s": stage_totals,
        "merge_stage_counts": stage_counts,
        "per_rank_total_s": per_rank_totals,
        "config": eta_config_to_jsonable(cfg),
    }
    with (output_dir / f"eta_timing_{run_tag}.json").open("w", encoding="utf-8") as f:
        json.dump(timing_merged, f, indent=2, ensure_ascii=False)

    print(f"[eta] merge complete -> {output_dir / f'eta_phase_diagram_{run_tag}.npz'}")
    return results


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eta phase diagram (CUDA); optional multi-GPU sharding + merge.")
    p.add_argument("--merge-only", action="store_true", help="Merge existing rank shards into final npz/figure.")
    p.add_argument("--rank", type=int, default=0, help="This process shard index [0, world_size).")
    p.add_argument("--world-size", type=int, default=1, help="Number of parallel shard processes.")
    p.add_argument("--resume", action="store_true", help="Skip compute if valid shard already exists (world_size>1).")
    p.add_argument("--run-tag", type=str, default=None, help="Override auto run tag (must match across ranks/merge).")
    p.add_argument("--shard-dir", type=str, default=None, help="Directory for shard npz/json (default: run output_dir).")
    p.add_argument("--data-phase-root", type=str, default=None, dest="data_phase_root")
    p.add_argument("--dtype", type=_str_to_dtype, default=None)
    p.add_argument("--linalg-library", type=str, choices=["default", "cusolver", "magma"], default=None)
    p.add_argument("--no-save-npz", action="store_true")
    p.add_argument("--no-save-fig", action="store_true")
    p.add_argument("--n-delta", type=int, default=None)
    p.add_argument("--n-q", type=int, default=None)
    p.add_argument("--n-k", type=int, default=None)
    p.add_argument("--kt-chunk", type=int, default=None)
    p.add_argument("--ja-chunk", type=int, default=None)
    p.add_argument("--delta-chunk", type=int, default=None)
    p.add_argument("--q-chunk", type=int, default=None)
    p.add_argument("--k-chunk", type=int, default=None)
    p.add_argument(
        "--max-eig-batch",
        type=int,
        default=None,
        help="Max number of 4x4 matrices per torch.linalg.eigvalsh call; <=0 disables splitting.",
    )
    p.add_argument("--profile", action="store_true", help="Enable cProfile for the main compute loop (per rank).")
    p.add_argument(
        "--print-run-tag",
        action="store_true",
        help="Print derived run_tag for current config and exit (no GPU work).",
    )
    return p.parse_args()


def _apply_overrides(cfg: EtaPhaseConfig, ns: argparse.Namespace) -> EtaPhaseConfig:
    if ns.data_phase_root is not None:
        cfg.data_phase_root = ns.data_phase_root
    if ns.dtype is not None:
        cfg.dtype = ns.dtype
    if ns.linalg_library is not None:
        cfg.linalg_library = ns.linalg_library  # type: ignore[assignment]
    if ns.no_save_npz:
        cfg.save_npz = False
    if ns.no_save_fig:
        cfg.save_fig = False
    if ns.n_delta is not None:
        cfg.n_delta = ns.n_delta
    if ns.n_q is not None:
        cfg.n_q = ns.n_q
    if ns.n_k is not None:
        cfg.n_k = ns.n_k
    if ns.kt_chunk is not None:
        cfg.kt_chunk = ns.kt_chunk
    if ns.ja_chunk is not None:
        cfg.ja_chunk = ns.ja_chunk
    if ns.delta_chunk is not None:
        cfg.delta_chunk = ns.delta_chunk
    if ns.q_chunk is not None:
        cfg.q_chunk = ns.q_chunk
    if ns.k_chunk is not None:
        cfg.k_chunk = ns.k_chunk
    if ns.max_eig_batch is not None:
        cfg.max_eig_batch = ns.max_eig_batch
    return cfg


def main() -> None:
    ns = _parse_cli()
    cfg = _apply_overrides(EtaPhaseConfig(), ns)
    shard_dir = Path(ns.shard_dir) if ns.shard_dir else None

    if ns.print_run_tag:
        c = cfg.scaled()
        kt_vec = build_kt_vec(c)
        ja_vec = build_ja_vec(c)
        q_vec = build_q_vec(c)
        tag = make_eta_run_tag(c, int(kt_vec.shape[0]), int(ja_vec.shape[0]), int(q_vec.shape[0]))
        print(tag)
        return

    if ns.world_size == 1 and ns.rank != 0:
        raise SystemExit("When --world-size is 1, --rank must be 0.")
    if ns.merge_only and ns.world_size < 2:
        raise SystemExit("--merge-only requires --world-size >= 2.")

    if ns.merge_only:
        if ns.run_tag is None:
            raise SystemExit("--merge-only requires --run-tag")
        merge_eta_phase_diagram_shards(cfg, ns.run_tag, world_size=ns.world_size, shard_dir=shard_dir)
        return

    run_eta_phase_diagram(
        cfg,
        rank=ns.rank,
        world_size=ns.world_size,
        resume=ns.resume,
        run_tag_override=ns.run_tag,
        shard_dir=shard_dir,
        profile=ns.profile,
    )


if __name__ == "__main__":
    main()
