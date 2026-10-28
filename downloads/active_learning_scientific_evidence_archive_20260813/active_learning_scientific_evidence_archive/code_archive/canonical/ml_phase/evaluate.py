from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .models import classification_accuracy, regression_rmse


@dataclass
class EvalMetrics:
    delta_rmse: float
    q_rmse: float
    eta_rmse: float
    ic_plus_rmse: float
    ic_minus_rmse: float
    phase_accuracy: float
    boundary_f1: float
    n_exact_calls: int
    estimated_reduction: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta_rmse": self.delta_rmse,
            "q_rmse": self.q_rmse,
            "eta_rmse": self.eta_rmse,
            "ic_plus_rmse": self.ic_plus_rmse,
            "ic_minus_rmse": self.ic_minus_rmse,
            "phase_accuracy": self.phase_accuracy,
            "boundary_f1": self.boundary_f1,
            "n_exact_calls": float(self.n_exact_calls),
            "estimated_reduction": self.estimated_reduction,
        }


def boundary_mask_from_labels(label_grid: np.ndarray) -> np.ndarray:
    label_grid = np.asarray(label_grid, dtype=np.int64)
    h, w = label_grid.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 1:] |= label_grid[:, 1:] != label_grid[:, :-1]
    mask[1:, :] |= label_grid[1:, :] != label_grid[:-1, :]
    return mask


def boundary_f1_with_tolerance(true_grid: np.ndarray, pred_grid: np.ndarray, tolerance_cells: int = 1) -> float:
    true_b = boundary_mask_from_labels(true_grid)
    pred_b = boundary_mask_from_labels(pred_grid)
    if tolerance_cells > 0:
        true_d = _binary_dilation_square(true_b, tolerance_cells)
        pred_d = _binary_dilation_square(pred_b, tolerance_cells)
    else:
        true_d = true_b
        pred_d = pred_b

    tp = np.sum(pred_b & true_d)
    fp = np.sum(pred_b & ~true_d)
    fn = np.sum(true_b & ~pred_d)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _binary_dilation_square(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        out[y0:y1, x0:x1] = True
    return out


def reconstruct_regular_grid(x: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values)
    kt_unique = np.unique(x[:, 0])
    ja_unique = np.unique(x[:, 1])
    n = kt_unique.size * ja_unique.size
    if n != x.shape[0]:
        return None

    kt_to_i = {v: i for i, v in enumerate(kt_unique)}
    ja_to_j = {v: j for j, v in enumerate(ja_unique)}
    grid = np.empty((ja_unique.size, kt_unique.size), dtype=values.dtype)
    seen = np.zeros(grid.shape, dtype=bool)
    for idx in range(x.shape[0]):
        i = kt_to_i[x[idx, 0]]
        j = ja_to_j[x[idx, 1]]
        grid[j, i] = values[idx]
        seen[j, i] = True
    if not np.all(seen):
        return None
    return grid


def estimate_exact_call_reduction(n_exact_calls: int, dense_grid_points: int) -> float:
    if dense_grid_points <= 0:
        return 0.0
    return float(dense_grid_points / max(n_exact_calls, 1))


def evaluate_predictions(
    x: np.ndarray,
    y_reg_true: np.ndarray,
    y_phase_true: np.ndarray,
    y_reg_pred: np.ndarray,
    y_phase_pred: np.ndarray,
    n_exact_calls: int,
    dense_grid_points: int,
) -> EvalMetrics:
    rmse = regression_rmse(y_reg_true, y_reg_pred)
    acc = classification_accuracy(y_phase_true, y_phase_pred)

    y_true_grid = reconstruct_regular_grid(x, y_phase_true)
    y_pred_grid = reconstruct_regular_grid(x, y_phase_pred)
    if y_true_grid is not None and y_pred_grid is not None:
        b_f1 = boundary_f1_with_tolerance(y_true_grid, y_pred_grid, tolerance_cells=1)
    else:
        b_f1 = float("nan")

    return EvalMetrics(
        delta_rmse=float(rmse[0]),
        q_rmse=float(rmse[1]),
        eta_rmse=float(rmse[2]),
        ic_plus_rmse=float(rmse[3]),
        ic_minus_rmse=float(rmse[4]),
        phase_accuracy=float(acc),
        boundary_f1=float(b_f1),
        n_exact_calls=int(n_exact_calls),
        estimated_reduction=estimate_exact_call_reduction(n_exact_calls, dense_grid_points),
    )
