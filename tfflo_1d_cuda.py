from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class SimConfig:
    delta0: float = 1.0
    t: float = 1.0
    lambda_ry: float = 0.6
    lambda_rz: float = 0.6
    u: float = 1.6
    mu: float = 0.55
    omega_d: float = 10.0
    kt: float = 0.01
    var_min: float = 0.4
    var_max: float = 1
    n_var: int = 40
    delta_min: float = 0.0
    delta_max: float = 0.6
    q_min: float = -0.8
    q_max: float = 0.0
    n_delta: int = 200
    n_q: int = 200
    n_k: int = 200
    nx: int = 200
    ja_chunk: int = 4
    q_chunk: int = 200
    # k_chunk: int = 160
    k_chunk: int = 200
    obc_eig_device: Literal["cpu", "cuda"] = "cuda"
    obc_batch: bool = True
    linalg_library: Literal["default", "cusolver", "magma"] = "magma"
    dtype: torch.dtype = torch.float32

    def scaled(self) -> "SimConfig":
        scale = self.delta0
        return SimConfig(
            delta0=scale,
            t=self.t * scale,
            lambda_ry=self.lambda_ry * scale,
            lambda_rz=self.lambda_rz * scale,
            u=self.u * scale,
            mu=self.mu,
            omega_d=self.omega_d,
            kt=self.kt,
            var_min=self.var_min,
            var_max=self.var_max,
            n_var=self.n_var,
            delta_min=self.delta_min,
            delta_max=self.delta_max,
            q_min=self.q_min,
            q_max=self.q_max,
            n_delta=self.n_delta,
            n_q=self.n_q,
            n_k=self.n_k,
            nx=self.nx,
            ja_chunk=self.ja_chunk,
            q_chunk=self.q_chunk,
            k_chunk=self.k_chunk,
            obc_eig_device=self.obc_eig_device,
            obc_batch=self.obc_batch,
            linalg_library=self.linalg_library,
            dtype=self.dtype,
        )


class StageTimer:
    def __init__(self) -> None:
        self._t0: float | None = None
        self.stage_totals_s: Dict[str, float] = {}
        self.stage_counts: Dict[str, int] = {}

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop_add(self, stage: str) -> None:
        if self._t0 is None:
            raise RuntimeError("Timer was not started.")
        dt = time.perf_counter() - self._t0
        self.stage_totals_s[stage] = self.stage_totals_s.get(stage, 0.0) + dt
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        self._t0 = None

    def add(self, stage: str, dt_s: float) -> None:
        self.stage_totals_s[stage] = self.stage_totals_s.get(stage, 0.0) + float(dt_s)
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for k in sorted(self.stage_totals_s.keys()):
            tot = self.stage_totals_s[k]
            cnt = self.stage_counts.get(k, 0)
            avg = tot / cnt if cnt else float("nan")
            lines.append(f"{k:>18s}: total={tot:9.3f}s  count={cnt:4d}  avg={avg:8.3f}s")
        return lines


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise TypeError(f"Unsupported dtype: {real_dtype}")


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this script but no GPU was detected.")
    return torch.device("cuda")

def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def build_k_grid(cfg: SimConfig, device: torch.device) -> torch.Tensor:
    return torch.linspace(-math.pi, math.pi, cfg.n_k, dtype=cfg.dtype, device=device)


def maybe_set_linalg_backend(cfg: SimConfig) -> None:
    if cfg.linalg_library == "default":
        return
    if torch.cuda.is_available():
        torch.backends.cuda.preferred_linalg_library(cfg.linalg_library)


def make_run_tag(cfg: SimConfig) -> str:
    dt = "fp32" if cfg.dtype == torch.float32 else "fp64" if cfg.dtype == torch.float64 else str(cfg.dtype).replace("torch.", "")
    obc = "obcGpu" if cfg.obc_eig_device == "cuda" else "obcCpu"
    obcb = "B1" if cfg.obc_batch else "B0"
    lib = f"lib{cfg.linalg_library}"
    return (
        f"nv{cfg.n_var}_nd{cfg.n_delta}_nq{cfg.n_q}_nk{cfg.n_k}_nx{cfg.nx}"
        f"_jc{cfg.ja_chunk}_qc{cfg.q_chunk}_kc{cfg.k_chunk}_{dt}_{obc}{obcb}_{lib}"
    )


def bdg_hamiltonian_batch(
    k: torch.Tensor,
    q: torch.Tensor,
    delta: torch.Tensor,
    t: float,
    lambda_ry: float,
    lambda_rz: float,
    ja: torch.Tensor | float,
    mu: torch.Tensor | float,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    ja_t = ja if isinstance(ja, torch.Tensor) else torch.as_tensor(ja, device=k.device, dtype=k.dtype)
    mu_t = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, device=k.device, dtype=k.dtype)
    k_b, q_b, delta_b, ja_b, mu_b = torch.broadcast_tensors(k, q, delta, ja_t, mu_t)
    kp = k_b + q_b / 2.0
    km = -k_b + q_b / 2.0

    h11 = (t + ja_b) * torch.cos(kp) + lambda_rz * torch.sin(kp) - mu_b
    h12 = -1j * lambda_ry * torch.sin(kp)
    h22 = (t - ja_b) * torch.cos(kp) - lambda_rz * torch.sin(kp) - mu_b

    h33 = -(t + ja_b) * torch.cos(km) - lambda_rz * torch.sin(km) + mu_b
    h34 = -1j * lambda_ry * torch.sin(km)
    h44 = -(t - ja_b) * torch.cos(km) + lambda_rz * torch.sin(km) + mu_b

    shape = k_b.shape + (4, 4)
    h = torch.zeros(shape, dtype=complex_dtype, device=k.device)
    h[..., 0, 0] = h11
    h[..., 0, 1] = h12
    h[..., 1, 0] = torch.conj(h12)
    h[..., 1, 1] = h22
    h[..., 2, 2] = h33
    h[..., 2, 3] = h34
    h[..., 3, 2] = torch.conj(h34)
    h[..., 3, 3] = h44

    h[..., 0, 3] = -delta_b
    h[..., 1, 2] = delta_b
    h[..., 2, 1] = delta_b
    h[..., 3, 0] = -delta_b
    return (h + torch.conj(h.transpose(-1, -2))) / 2.0

def fermi_weighted_sum(evals: torch.Tensor, kt: float) -> torch.Tensor:
    # Numerically stable form of E / (exp(E/kT) + 1).
    return torch.sum(evals * torch.sigmoid(-evals / kt), dim=-1)

def compute_omega_for_ja_batch(
    ja_vec: torch.Tensor,
    cfg: SimConfig,
    k_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = k_vec.device
    ctype = complex_dtype_for(cfg.dtype)
    delta_vec = torch.linspace(cfg.delta_min, cfg.delta_max, cfg.n_delta, dtype=cfg.dtype, device=device)
    q_vec = torch.linspace(cfg.q_min, cfg.q_max, cfg.n_q, dtype=cfg.dtype, device=device)

    mask = torch.abs(cfg.t * torch.cos(k_vec) - cfg.mu) < cfg.omega_d
    valid_k = k_vec[mask]
    if valid_k.numel() == 0:
        raise RuntimeError("No valid k-point satisfies Debye cutoff.")

    omega_mat = torch.empty((ja_vec.numel(), cfg.n_delta, cfg.n_q), dtype=cfg.dtype, device=device)
    delta_term = (delta_vec ** 2 / cfg.u).view(1, cfg.n_delta, 1)
    nk = float(cfg.n_k)

    for q_start in range(0, cfg.n_q, cfg.q_chunk):
        q_end = min(q_start + cfg.q_chunk, cfg.n_q)
        q_chunk = q_vec[q_start:q_end]

        d_b = delta_vec.view(1, cfg.n_delta, 1, 1)
        q_b = q_chunk.view(1, 1, q_chunk.numel(), 1)
        ja_b = ja_vec.view(ja_vec.numel(), 1, 1, 1)
        int_sum = torch.zeros((ja_vec.numel(), cfg.n_delta, q_chunk.numel()), dtype=cfg.dtype, device=device)
        int_sum0 = torch.zeros((ja_vec.numel(), cfg.n_delta, q_chunk.numel()), dtype=cfg.dtype, device=device)
        for k_start in range(0, valid_k.numel(), cfg.k_chunk):
            k_end = min(k_start + cfg.k_chunk, valid_k.numel())
            k_part = valid_k[k_start:k_end]
            k_b = k_part.view(1, 1, 1, k_part.numel())

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
            evals = torch.linalg.eigvalsh(h).real
            int_sum += torch.sum(fermi_weighted_sum(evals, cfg.kt), dim=-1)

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
            evals0 = torch.linalg.eigvalsh(h0).real
            int_sum0 += torch.sum(fermi_weighted_sum(evals0, cfg.kt), dim=-1)

        omega_chunk = 0.5 * int_sum / nk + delta_term - 0.5 * int_sum0 / nk
        omega_mat[:, :, q_start:q_end] = omega_chunk

    return omega_mat, delta_vec, q_vec, valid_k


def compute_z2(cfg: SimConfig, ja: float, q_opt: float, delta_opt: float) -> float:
    lhs = (
        (cfg.mu + cfg.t * math.cos(q_opt / 2.0)) ** 2
        - (ja * math.cos(q_opt / 2.0) + cfg.lambda_rz * math.sin(q_opt / 2.0)) ** 2
        + delta_opt**2
        - (cfg.lambda_ry * math.sin(q_opt / 2.0)) ** 2
    )
    rhs = (
        (cfg.mu - cfg.t * math.cos(q_opt / 2.0)) ** 2
        - (ja * math.cos(q_opt / 2.0) + cfg.lambda_rz * math.sin(q_opt / 2.0)) ** 2
        + delta_opt**2
        - (cfg.lambda_ry * math.sin(q_opt / 2.0)) ** 2
    )
    return float(np.sign(lhs * rhs))


def build_obc_hamiltonian(cfg: SimConfig, ja: float, delta_opt: float, q_opt: float) -> np.ndarray:
    sigma_0 = np.eye(2, dtype=np.complex64)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
    tau_0 = np.eye(2, dtype=np.complex64)
    tau_x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    tau_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    tau_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

    n_states = 4 * cfg.nx
    h_const = np.zeros((n_states, n_states), dtype=np.complex64)

    hop = (
        0.5 * cfg.t * np.kron(tau_z, sigma_0)
        + 0.5 * ja * np.kron(tau_0, sigma_z)
        - 0.5j * cfg.lambda_rz * np.kron(tau_z, sigma_z)
        + 0.5j * cfg.lambda_ry * np.kron(tau_z, sigma_y)
    )
    for i in range(cfg.nx - 1):
        idx_i = slice(i * 4, (i + 1) * 4)
        idx_j = slice((i + 1) * 4, (i + 2) * 4)
        h_const[idx_i, idx_j] += hop
        h_const[idx_j, idx_i] += hop.conj().T

    for i in range(cfg.nx):
        delta_onsite = (
            delta_opt * math.cos(q_opt * (i + 1)) * np.kron(tau_x, sigma_0)
            - delta_opt * math.sin(q_opt * (i + 1)) * np.kron(tau_y, sigma_0)
        )
        idx = slice(i * 4, (i + 1) * 4)
        h_const[idx, idx] += delta_onsite

    h_mu = np.zeros((n_states, n_states), dtype=np.complex64)
    mu_term = -cfg.mu * np.kron(tau_z, sigma_0)
    for i in range(cfg.nx):
        idx = slice(i * 4, (i + 1) * 4)
        h_mu[idx, idx] += mu_term

    h = h_const + h_mu
    return (h + np.conj(h.T)) / 2.0


def bdg_hamiltonian_numpy(
    k: float,
    q: float,
    delta: float,
    t: float,
    lambda_ry: float,
    lambda_rz: float,
    ja: float,
    mu: float,
) -> np.ndarray:
    kp = k + q / 2.0
    km = -k + q / 2.0
    h11 = (t + ja) * math.cos(kp) + lambda_rz * math.sin(kp) - mu
    h12 = -1j * lambda_ry * math.sin(kp)
    h22 = (t - ja) * math.cos(kp) - lambda_rz * math.sin(kp) - mu
    h33 = -(t + ja) * math.cos(km) - lambda_rz * math.sin(km) + mu
    h34 = -1j * lambda_ry * math.sin(km)
    h44 = -(t - ja) * math.cos(km) + lambda_rz * math.sin(km) + mu
    h = np.zeros((4, 4), dtype=np.complex64)
    h[0, 0], h[0, 1], h[1, 0], h[1, 1] = h11, h12, np.conj(h12), h22
    h[2, 2], h[2, 3], h[3, 2], h[3, 3] = h33, h34, np.conj(h34), h44
    h[0, 3], h[1, 2], h[2, 1], h[3, 0] = -delta, delta, delta, -delta
    return (h + np.conj(h.T)) / 2.0


def plot_basic(
    cfg: SimConfig,
    var_vec: np.ndarray,
    delta_opt_vec: np.ndarray,
    q_opt_vec: np.ndarray,
    e_vals: np.ndarray,
    z2_vec: np.ndarray,
    out_dir: Path,
) -> None:
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(var_vec, delta_opt_vec, "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("JA")
    ax1.set_ylabel(r"$\Delta_{opt}$")
    ax1.set_title("Optimal Pairing Gap vs Chemical Potential")
    ax1.grid(True)
    ax1.set_xlim(cfg.var_min, cfg.var_max)
    ax1.set_ylim(0, max(delta_opt_vec.max() * 1.1, 1e-6))

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(var_vec, q_opt_vec, "r-o", linewidth=2, markersize=6)
    ax2.set_xlabel("JA")
    ax2.set_ylabel(r"$q_{opt}$ (radians)")
    ax2.set_title("Optimal Center-of-Mass Momentum vs Chemical Potential")
    ax2.grid(True)
    ax2.set_xlim(cfg.var_min, cfg.var_max)

    ax3 = fig.add_subplot(2, 2, 3)
    for i in range(e_vals.shape[1]):
        ax3.plot(var_vec, e_vals[:, i], "bo", markersize=1.5)
    ax3.set_xlabel("JA")
    ax3.set_ylabel("E")
    ax3.set_title("spectrum vs Chemical Potential")
    ax3.grid(True)
    ax3.set_xlim(cfg.var_min, cfg.var_max)
    ax3.set_ylim(-0.1, 0.1)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(var_vec, z2_vec, "m-o", linewidth=2, markersize=6)
    ax4.set_xlabel("JA")
    ax4.set_ylabel("Z2 Invariant")
    ax4.set_title("Topological Invariant vs Chemical Potential")
    ax4.grid(True)
    ax4.set_xlim(cfg.var_min, cfg.var_max)
    ax4.set_ylim(-1.5, 1.5)
    ax4.axhline(0, color="k", linestyle="--")
    ax4.axhline(1, color="r", linestyle="--")
    ax4.axhline(-1, color="b", linestyle="--")

    fig.tight_layout()
    fig.savefig(out_dir / "basic_results.png", dpi=200)
    fig.savefig(out_dir / "basic_results.pdf")
    plt.close(fig)


def plot_prl_style(
    cfg: SimConfig,
    var_vec: np.ndarray,
    delta_opt_vec: np.ndarray,
    q_opt_vec: np.ndarray,
    e_vals: np.ndarray,
    out_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "DejaVu Sans",
        }
    )
    fig = plt.figure(figsize=(10, 6))
    var_samples = np.array([0.50, 0.593, 0.7], dtype=np.float32)
    cfflo_max = 0.593

    ax = fig.add_subplot(2, 2, 1)
    ax_r = ax.twinx()
    l1 = ax.plot(var_vec, delta_opt_vec, "b-o", linewidth=1.2, label=r"$\Delta_{opt}$")[0]
    l2 = ax_r.plot(var_vec, q_opt_vec, "r-s", linewidth=1.2, label=r"$q_{opt}$")[0]

    for s in var_samples:
        d_s = np.interp(s, var_vec, delta_opt_vec)
        q_s = np.interp(s, var_vec, q_opt_vec)
        ax.plot([s], [d_s], "ko", markersize=5)
        ax_r.plot([s], [q_s], "ko", markersize=5)
        ax.axvline(s, color="k", linestyle="--", linewidth=0.5)

    ax.set_xlabel(r"$J_A / t$", fontsize=14)
    ax.set_ylabel(r"$\Delta_{opt} / t$", color="b", fontsize=14)
    ax_r.set_ylabel(r"$q_{opt}$ (1/a)", color="r", fontsize=14)
    ax.set_xlim(cfg.var_min, cfg.var_max)
    ax.legend([l1, l2], [r"$\Delta_{opt}$", r"$q_{opt}$"], loc="upper right", frameon=False, fontsize=12)
    ax.grid(True, alpha=0.2)

    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(e_vals.shape[1]):
        ax2.plot(var_vec, e_vals[:, i], ".", color=(0.6, 0.6, 0.6), markersize=2)

    min_abs = np.min(np.abs(e_vals), axis=1)
    tol = 0.005
    for idx, ja in enumerate(var_vec):
        if ja > cfflo_max:
            near_idx = np.where(np.abs(np.abs(e_vals[idx, :]) - min_abs[idx]) < tol)[0]
            for j in near_idx:
                ax2.plot(ja, e_vals[idx, j], "r.", markersize=8)

    y_fill = [-0.1, -0.1, 0.1, 0.1]
    ax2.fill([cfg.var_min, cfflo_max, cfflo_max, cfg.var_min], y_fill, color=(0.85, 0.95, 0.85), alpha=0.2, ec="none")
    ax2.fill([cfflo_max, cfg.var_max, cfg.var_max, cfflo_max], y_fill, color=(0.95, 0.85, 0.95), alpha=0.2, ec="none")
    ax2.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax2.axvline(cfflo_max, color="k", linewidth=0.8)
    ax2.text(cfg.var_min + 0.02, 0.085, "cFFLO", fontsize=12, color=(0, 0.5, 0), fontweight="bold")
    ax2.text(cfflo_max + 0.02, 0.085, "tFFLO", fontsize=12, color=(0.5, 0, 0.5), fontweight="bold")
    ax2.set_xlabel(r"$J_A / t$", fontsize=14)
    ax2.set_ylabel(r"$E / t$", fontsize=14)
    ax2.set_xlim(cfg.var_min, cfg.var_max)
    ax2.set_ylim(-0.15, 0.15)

    k_plot = np.linspace(-math.pi, math.pi, 500, dtype=np.float32)
    for i, s in enumerate(var_samples):
        idx = int(np.argmin(np.abs(var_vec - s)))
        d_s = float(delta_opt_vec[idx])
        q_s = float(q_opt_vec[idx])
        energies = np.zeros((k_plot.size, 4), dtype=np.float32)
        for ik, kval in enumerate(k_plot):
            ek = np.sort(np.real(np.linalg.eigvals(bdg_hamiltonian_numpy(kval, q_s, d_s, cfg.t, cfg.lambda_ry, cfg.lambda_rz, float(s), cfg.mu))))
            energies[ik, :] = ek

        axb = fig.add_subplot(2, 3, 4 + i)
        for b in range(4):
            axb.plot(k_plot, energies[:, b], "k-", linewidth=0.7)
        axb.axhline(0, color="k", linestyle="--", linewidth=0.5)
        if i == 0:
            axb.set_ylabel(r"$E / t$", fontsize=14)
        axb.set_xlabel("k (1/a)", fontsize=14)
        axb.set_title(f"J_A/t = {s:.2f}", fontsize=14)
        axb.text(0.05, 0.88, rf"$\Delta={d_s:.3f}$", transform=axb.transAxes, fontsize=10)
        axb.text(0.05, 0.80, f"q={q_s:.3f}", transform=axb.transAxes, fontsize=10)
        axb.set_xlim(-math.pi, math.pi)
        axb.set_ylim(-1.5, 1.5)
        axb.set_xticks([-math.pi, 0, math.pi])
        axb.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.08, wspace=0.35, hspace=0.35)
    fig.savefig(out_dir / "prl_combined_figure.png", dpi=250)
    fig.savefig(out_dir / "prl_combined_figure.pdf")
    plt.close(fig)


def run(cfg: SimConfig) -> Dict[str, np.ndarray]:
    cfg = cfg.scaled()
    device = require_cuda()
    maybe_set_linalg_backend(cfg)
    run_tag = make_run_tag(cfg)
    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(exist_ok=True)

    var_vec = np.linspace(cfg.var_min, cfg.var_max, cfg.n_var, dtype=np.float32)
    delta_opt_vec = np.zeros(cfg.n_var, dtype=np.float32)
    q_opt_vec = np.zeros(cfg.n_var, dtype=np.float32)
    omega_min_vec = np.zeros(cfg.n_var, dtype=np.float32)
    z2_vec = np.zeros(cfg.n_var, dtype=np.float32)
    n_states = 4 * cfg.nx
    e_vals = np.zeros((cfg.n_var, n_states), dtype=np.float32)

    timer = StageTimer()
    t0 = time.perf_counter()
    k_vec = build_k_grid(cfg, device)
    synchronize_if_cuda(device)
    t1 = time.perf_counter()

    print("Scanning over JA...")
    print(f"JA range: [{cfg.var_min:.2f}, {cfg.var_max:.2f}]")
    print("Progress:")
    omega_elapsed = 0.0
    obc_elapsed = 0.0

    ja_all = torch.as_tensor(var_vec, device=device, dtype=cfg.dtype)
    delta_vec_t: torch.Tensor | None = None
    q_vec_t: torch.Tensor | None = None

    for ja_start in range(0, cfg.n_var, cfg.ja_chunk):
        ja_end = min(ja_start + cfg.ja_chunk, cfg.n_var)
        ja_part = ja_all[ja_start:ja_end]

        scan_start = time.perf_counter()
        omega_part, delta_vec_t, q_vec_t, _valid_k = compute_omega_for_ja_batch(ja_part, cfg, k_vec)
        synchronize_if_cuda(device)
        scan_end = time.perf_counter()
        dt = scan_end - scan_start
        omega_elapsed += dt
        timer.add("omega_ja_chunk", dt)

        omega_flat = omega_part.reshape(ja_part.numel(), -1)
        argmin = torch.argmin(omega_flat, dim=1)
        i_delta = (argmin // cfg.n_q).to(torch.int64)
        i_q = (argmin % cfg.n_q).to(torch.int64)

        delta_opt_part = delta_vec_t[i_delta].to("cpu", non_blocking=True).numpy()
        q_opt_part = q_vec_t[i_q].to("cpu", non_blocking=True).numpy()
        omega_min_part = omega_flat.gather(1, argmin[:, None]).squeeze(1).to("cpu", non_blocking=True).numpy()

        delta_opt_vec[ja_start:ja_end] = delta_opt_part.astype(np.float32)
        q_opt_vec[ja_start:ja_end] = q_opt_part.astype(np.float32)
        omega_min_vec[ja_start:ja_end] = omega_min_part.astype(np.float32)

    if delta_vec_t is None or q_vec_t is None:
        raise RuntimeError("Internal error: delta/q grids were not initialized.")

    obc_use_cuda = cfg.obc_eig_device == "cuda"
    obc_device = device if obc_use_cuda else torch.device("cpu")
    obc_ctype = complex_dtype_for(torch.float32)

    for i, ja in enumerate(var_vec):
        z2_vec[i] = compute_z2(cfg, float(ja), float(q_opt_vec[i]), float(delta_opt_vec[i]))

    if obc_use_cuda and cfg.obc_batch:
        obc_start = time.perf_counter()
        h_stack = np.stack(
            [build_obc_hamiltonian(cfg, float(ja), float(delta_opt_vec[i]), float(q_opt_vec[i])) for i, ja in enumerate(var_vec)],
            axis=0,
        )
        timer.add("obc_build_stack", time.perf_counter() - obc_start)

        obc_start = time.perf_counter()
        h_t = torch.from_numpy(h_stack).to(device=obc_device, dtype=obc_ctype, non_blocking=True)
        ev = torch.linalg.eigvals(h_t).real.to("cpu", non_blocking=True).numpy()
        for i in range(cfg.n_var):
            e_vals[i, :] = np.sort(ev[i, :].astype(np.float32))
        synchronize_if_cuda(device)
        obc_end = time.perf_counter()
        obc_elapsed += obc_end - obc_start
        timer.add("obc_eig_batched", obc_end - obc_start)
        for i, ja in enumerate(var_vec):
            print(
                f"JA = {ja:.3f}: Delta_opt = {float(delta_opt_vec[i]):.4f}, q_opt = {float(q_opt_vec[i]):.4f}, Omega_min = {float(omega_min_vec[i]):.4f}"
            )
    else:
        for i, ja in enumerate(var_vec):
            delta_opt = float(delta_opt_vec[i])
            q_opt = float(q_opt_vec[i])
            min_omega = float(omega_min_vec[i])

            obc_start = time.perf_counter()
            h_total = build_obc_hamiltonian(cfg, float(ja), delta_opt, q_opt)
            timer.add("obc_build_one", time.perf_counter() - obc_start)

            obc_start = time.perf_counter()
            if obc_use_cuda:
                h_t = torch.from_numpy(h_total).to(device=obc_device, dtype=obc_ctype, non_blocking=True)
                ev = torch.linalg.eigvals(h_t).real.to("cpu", non_blocking=True).numpy()
            else:
                ev = np.real(np.linalg.eigvals(h_total))
            e_vals[i, :] = np.sort(ev.astype(np.float32))
            synchronize_if_cuda(device)
            obc_end = time.perf_counter()
            obc_elapsed += obc_end - obc_start
            timer.add("obc_eig_one", obc_end - obc_start)

            print(f"JA = {ja:.3f}: Delta_opt = {delta_opt:.4f}, q_opt = {q_opt:.4f}, Omega_min = {min_omega:.4f}")

    plot_start = time.perf_counter()
    plot_subdir = out_dir / run_tag
    plot_subdir.mkdir(exist_ok=True)
    plot_basic(cfg, var_vec, delta_opt_vec, q_opt_vec, e_vals, z2_vec, plot_subdir)
    plot_prl_style(cfg, var_vec, delta_opt_vec, q_opt_vec, e_vals, plot_subdir)
    plot_end = time.perf_counter()
    timer.add("plotting", plot_end - plot_start)

    print("\n=== Timing (seconds) ===")
    print(f"k-grid init: {t1 - t0:.3f}")
    print(f"Omega scan total: {omega_elapsed:.3f}")
    print(f"OBC spectrum total: {obc_elapsed:.3f}")
    print(f"Plotting total: {plot_end - plot_start:.3f}")
    print(f"All stages total: {plot_end - t0:.3f}")
    print("\n=== Stage timing details ===")
    for line in timer.summary_lines():
        print(line)

    timing_path = Path(__file__).parent / f"timing_{run_tag}.json"
    with timing_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_tag": run_tag,
                "k_grid_init_s": float(t1 - t0),
                "omega_scan_total_s": float(omega_elapsed),
                "obc_total_s": float(obc_elapsed),
                "plotting_s": float(plot_end - plot_start),
                "all_total_s": float(plot_end - t0),
                "stage_totals_s": timer.stage_totals_s,
                "stage_counts": timer.stage_counts,
                "config": {
                    "n_var": cfg.n_var,
                    "n_delta": cfg.n_delta,
                    "n_q": cfg.n_q,
                    "n_k": cfg.n_k,
                    "nx": cfg.nx,
                    "ja_chunk": cfg.ja_chunk,
                    "q_chunk": cfg.q_chunk,
                    "k_chunk": cfg.k_chunk,
                    "dtype": str(cfg.dtype),
                    "obc_eig_device": cfg.obc_eig_device,
                    "obc_batch": cfg.obc_batch,
                    "linalg_library": cfg.linalg_library,
                    "var_min": cfg.var_min,
                    "var_max": cfg.var_max,
                    "mu": cfg.mu,
                    "u": cfg.u,
                    "omega_d": cfg.omega_d,
                    "kt": cfg.kt,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "JA": var_vec,
        "Delta_opt_vec": delta_opt_vec,
        "q_opt_vec": q_opt_vec,
        "Omega_min_vec": omega_min_vec,
        "Z2_vec": z2_vec,
        "E_vals": e_vals,
    }


if __name__ == "__main__":
    results = run(SimConfig())
    cfg = SimConfig().scaled()
    tag = make_run_tag(cfg)
    np.savez(Path(__file__).parent / f"results_tfflo_1d_cuda_{tag}.npz", **results)
