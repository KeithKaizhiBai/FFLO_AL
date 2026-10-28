from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from tfflo_1d_cuda import bdg_hamiltonian_batch


Backend = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class TopologyModelParams:
    t: float = 1.0
    lambda_ry: float = 0.6
    lambda_rz: float = 0.6
    mu: float = 0.55

    def energy_scale(self, ja: np.ndarray | float, delta: np.ndarray | float) -> np.ndarray:
        ja_arr = np.asarray(ja, dtype=np.float64)
        delta_arr = np.asarray(delta, dtype=np.float64)
        scale = np.maximum.reduce(
            [
                np.full_like(ja_arr, abs(self.t), dtype=np.float64),
                np.full_like(ja_arr, abs(self.lambda_ry), dtype=np.float64),
                np.full_like(ja_arr, abs(self.lambda_rz), dtype=np.float64),
                np.full_like(ja_arr, abs(self.mu), dtype=np.float64),
                np.abs(ja_arr),
                np.abs(delta_arr),
                np.full_like(ja_arr, 1.0, dtype=np.float64),
            ]
        )
        return scale


def pfaffian_4x4(a: np.ndarray) -> complex:
    """Return the Pfaffian of a 4x4 antisymmetric matrix."""
    return a[0, 1] * a[2, 3] - a[0, 2] * a[1, 3] + a[0, 3] * a[1, 2]


def nambu_to_majorana_transform() -> np.ndarray:
    """Map Majorana operators to the project's Nambu basis.

    The project Nambu order is
    (c_up, c_down, c_up^dagger, c_down^dagger).  The Majorana order used here is
    (gamma_up_1, gamma_up_2, gamma_down_1, gamma_down_2), with
    c = (gamma_1 + i gamma_2) / 2.
    """
    w = np.zeros((4, 4), dtype=np.complex128)
    w[0, 0] = 0.5
    w[0, 1] = 0.5j
    w[1, 2] = 0.5
    w[1, 3] = 0.5j
    w[2, 0] = 0.5
    w[2, 1] = -0.5j
    w[3, 2] = 0.5
    w[3, 3] = -0.5j
    return w


def majorana_antisymmetric_from_bdg(h_bdg: np.ndarray) -> np.ndarray:
    """Convert a 4x4 BdG Hamiltonian to the Majorana antisymmetric matrix."""
    w = nambu_to_majorana_transform()
    m = w.conj().T @ np.asarray(h_bdg, dtype=np.complex128) @ w
    a = -1j * (m - m.T)
    return np.asarray(a, dtype=np.complex128)


def bdg_hamiltonian_numpy_from_project_builder(
    k: float,
    q: float,
    delta: float,
    ja: float,
    params: TopologyModelParams,
) -> np.ndarray:
    """Build one Hamiltonian through the same project BdG batch builder."""
    k_t = torch.as_tensor(k, dtype=torch.float64)
    q_t = torch.as_tensor(q, dtype=torch.float64)
    d_t = torch.as_tensor(delta, dtype=torch.float64)
    ja_t = torch.as_tensor(ja, dtype=torch.float64)
    h = bdg_hamiltonian_batch(
        k=k_t,
        q=q_t,
        delta=d_t,
        t=float(params.t),
        lambda_ry=float(params.lambda_ry),
        lambda_rz=float(params.lambda_rz),
        ja=ja_t,
        mu=float(params.mu),
        complex_dtype=torch.complex128,
    )
    return h.detach().cpu().numpy().astype(np.complex128)


class TopologyPfaffianOracle:
    """Pfaffian Z2 diagnostics for the existing BdG Hamiltonian convention."""

    def __init__(self, params: TopologyModelParams | None = None, pf_tol_rel: float = 1e-8) -> None:
        self.params = params or TopologyModelParams()
        self.pf_tol_rel = float(pf_tol_rel)

    def analytic_pfaffians(
        self,
        delta: np.ndarray | float,
        q: np.ndarray | float,
        ja: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return P0, Ppi, product, and dimensionless margin.

        The k labels are fixed by direct comparison against the Majorana-basis
        matrix built from the project BdG Hamiltonian: k=0 corresponds to the
        (mu - t cos(q/2)) branch, and k=pi to the (mu + t cos(q/2)) branch.
        """
        delta_a = np.asarray(delta, dtype=np.float64).copy()
        q_a = np.asarray(q, dtype=np.float64).copy()
        ja_a = np.asarray(ja, dtype=np.float64).copy()
        c = np.cos(q_a / 2.0)
        s = np.sin(q_a / 2.0)
        d2 = (ja_a * c + self.params.lambda_rz * s) ** 2
        y2 = (self.params.lambda_ry * s) ** 2
        p0 = (self.params.mu - self.params.t * c) ** 2 + delta_a**2 - y2 - d2
        ppi = (self.params.mu + self.params.t * c) ** 2 + delta_a**2 - y2 - d2
        product = p0 * ppi
        e_scale = self.params.energy_scale(ja_a, delta_a)
        margin = np.minimum(np.abs(p0), np.abs(ppi)) / np.maximum(e_scale**2, 1e-300)
        return p0, ppi, product, margin

    def z2_status(
        self,
        delta: np.ndarray,
        q: np.ndarray,
        ja: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0, ppi, product, margin = self.analytic_pfaffians(delta, q, ja)
        e_scale = self.params.energy_scale(ja, delta)
        pf_tol = self.pf_tol_rel * np.maximum(e_scale**2, 1e-300)
        boundary = (np.abs(p0) <= pf_tol) | (np.abs(ppi) <= pf_tol)
        z2 = np.full(product.shape, -1, dtype=np.int64)
        z2[(product < 0) & ~boundary] = 1
        z2[(product > 0) & ~boundary] = 0
        return p0, ppi, product, margin, z2

    def numeric_pfaffians(self, delta: float, q: float, ja: float) -> tuple[complex, complex, float, float]:
        pfs: list[complex] = []
        antisym_errors: list[float] = []
        imag_maxes: list[float] = []
        for k in (0.0, math.pi):
            h = bdg_hamiltonian_numpy_from_project_builder(k, q, delta, ja, self.params)
            a = majorana_antisymmetric_from_bdg(h)
            denom = max(float(np.linalg.norm(a)), 1e-300)
            antisym_errors.append(float(np.linalg.norm(a + a.T) / denom))
            imag_maxes.append(float(np.max(np.abs(np.imag(a)))))
            pfs.append(pfaffian_4x4(a))
        return pfs[0], pfs[1], max(antisym_errors), max(imag_maxes)


class BulkGapOracle:
    """Chunked full-Brillouin-zone quasiparticle-gap scanner."""

    def __init__(
        self,
        params: TopologyModelParams | None = None,
        backend: Backend = "cpu",
        point_chunk: int = 64,
        k_chunk: int = 512,
    ) -> None:
        self.params = params or TopologyModelParams()
        self.backend = backend
        self.point_chunk = int(point_chunk)
        self.k_chunk = int(k_chunk)

    def device(self) -> torch.device:
        if self.backend == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU backend requested but torch.cuda.is_available() is false")
            return torch.device("cuda")
        return torch.device("cpu")

    def gap_at_k(self, delta: float, q: float, ja: float, k: float) -> float:
        """Evaluate the minimum absolute BdG eigenvalue at one momentum."""
        h = bdg_hamiltonian_numpy_from_project_builder(
            k=float(k),
            q=float(q),
            delta=float(delta),
            ja=float(ja),
            params=self.params,
        )
        eig = np.linalg.eigvalsh(h)
        return float(np.min(np.abs(eig)))

    def compute(
        self,
        delta: np.ndarray,
        q: np.ndarray,
        ja: np.ndarray,
        nk: int,
    ) -> dict[str, object]:
        device = self.device()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        delta_a = np.asarray(delta, dtype=np.float64).copy()
        q_a = np.asarray(q, dtype=np.float64).copy()
        ja_a = np.asarray(ja, dtype=np.float64).copy()
        n = int(delta_a.size)
        gaps = np.full(n, np.inf, dtype=np.float64)
        k_at = np.full(n, np.nan, dtype=np.float64)
        k_grid = torch.linspace(-math.pi, math.pi, int(nk), dtype=torch.float64, device=device)
        for p0 in range(0, n, self.point_chunk):
            p1 = min(p0 + self.point_chunk, n)
            d_t = torch.as_tensor(delta_a[p0:p1], dtype=torch.float64, device=device).view(-1, 1)
            q_t = torch.as_tensor(q_a[p0:p1], dtype=torch.float64, device=device).view(-1, 1)
            ja_t = torch.as_tensor(ja_a[p0:p1], dtype=torch.float64, device=device).view(-1, 1)
            best_gap = torch.full((p1 - p0,), torch.inf, dtype=torch.float64, device=device)
            best_k = torch.full((p1 - p0,), torch.nan, dtype=torch.float64, device=device)
            for k0 in range(0, int(nk), self.k_chunk):
                k1 = min(k0 + self.k_chunk, int(nk))
                k_t = k_grid[k0:k1].view(1, -1)
                h = bdg_hamiltonian_batch(
                    k=k_t,
                    q=q_t,
                    delta=d_t,
                    t=float(self.params.t),
                    lambda_ry=float(self.params.lambda_ry),
                    lambda_rz=float(self.params.lambda_rz),
                    ja=ja_t,
                    mu=float(self.params.mu),
                    complex_dtype=torch.complex128,
                )
                evals = torch.linalg.eigvalsh(h).real
                gap_by_k = torch.amin(torch.abs(evals), dim=-1)
                local_gap, local_idx = torch.min(gap_by_k, dim=1)
                better = local_gap < best_gap
                if torch.any(better):
                    best_gap = torch.where(better, local_gap, best_gap)
                    best_k = torch.where(better, k_t.reshape(-1)[local_idx], best_k)
                del h, evals, gap_by_k, local_gap, local_idx, better
            gaps[p0:p1] = best_gap.detach().cpu().numpy()
            k_at[p0:p1] = best_k.detach().cpu().numpy()
            del d_t, q_t, ja_t, best_gap, best_k
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_vram = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
        else:
            peak_vram = float("nan")
        elapsed = float(time.perf_counter() - t0)
        return {
            "bulk_gap": gaps,
            "k_at_bulk_gap": k_at,
            "runtime_seconds": elapsed,
            "backend": self.backend,
            "nk": int(nk),
            "point_count": n,
            "k_hamiltonians": int(n) * int(nk),
            "points_per_second": float(n / elapsed) if elapsed > 0 else float("inf"),
            "k_hamiltonians_per_second": float((int(n) * int(nk)) / elapsed) if elapsed > 0 else float("inf"),
            "peak_vram_mb": peak_vram,
        }
