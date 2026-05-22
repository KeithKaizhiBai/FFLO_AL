from __future__ import annotations

import numpy as np


PHASE_NORMAL = 0
PHASE_UNIFORM_SC = 1
PHASE_FFLO = 2

PHASE_NAMES = {
    PHASE_NORMAL: "normal",
    PHASE_UNIFORM_SC: "uniform_SC",
    PHASE_FFLO: "FFLO",
}


def phase_label(delta_opt: np.ndarray, q_opt: np.ndarray, delta_eps: float, q_eps: float) -> np.ndarray:
    delta_opt = np.asarray(delta_opt, dtype=np.float64)
    q_opt = np.asarray(q_opt, dtype=np.float64)
    out = np.full(delta_opt.shape, PHASE_FFLO, dtype=np.int64)
    out[delta_opt < delta_eps] = PHASE_NORMAL
    uniform_mask = (delta_opt >= delta_eps) & (np.abs(q_opt) < q_eps)
    out[uniform_mask] = PHASE_UNIFORM_SC
    return out


def eta_sign_label(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    out = np.zeros(eta.shape, dtype=np.int64)
    out[eta > 0.0] = 1
    out[eta < 0.0] = -1
    return out


def strong_diode_label(eta: np.ndarray, eta_strong: float) -> np.ndarray:
    eta = np.asarray(eta, dtype=np.float64)
    return (np.abs(eta) > eta_strong).astype(np.int64)


def label_name_map(values: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    return np.vectorize(mapping.get, otypes=[object])(values)

