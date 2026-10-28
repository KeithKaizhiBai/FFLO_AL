from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ActiveLearningConfig


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_std(x: np.ndarray, axis: int = 0) -> np.ndarray:
    s = np.std(x, axis=axis, keepdims=True)
    s[s < 1e-12] = 1.0
    return s


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def fit_scaler(x: np.ndarray) -> StandardScaler:
    mean = np.mean(x, axis=0, keepdims=True)
    std = _safe_std(x, axis=0)
    return StandardScaler(mean=mean, std=std)


def split_train_val(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_phase: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    indices = np.arange(n)

    # Stratify by phase label to keep boundary samples represented in val.
    train_idx_parts: List[np.ndarray] = []
    val_idx_parts: List[np.ndarray] = []
    for cls in np.unique(y_phase):
        cls_idx = indices[y_phase == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(val_fraction * cls_idx.size)))
        n_val = min(n_val, cls_idx.size - 1) if cls_idx.size > 1 else 1
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=np.int64)
    if train_idx.size == 0:
        train_idx = indices[:-1]
        val_idx = indices[-1:]

    return {
        "x_train": x[train_idx],
        "y_reg_train": y_reg[train_idx],
        "y_phase_train": y_phase[train_idx],
        "x_val": x[val_idx],
        "y_reg_val": y_reg[val_idx],
        "y_phase_val": y_phase[val_idx],
    }


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchEnsembleRegressor:
    def __init__(self, cfg: ActiveLearningConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.members: List[_MLP] = []
        self.x_scaler: StandardScaler | None = None
        self.y_scaler: StandardScaler | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.x_scaler = fit_scaler(x)
        self.y_scaler = fit_scaler(y)
        x_s = self.x_scaler.transform(x).astype(np.float32)
        y_s = self.y_scaler.transform(y).astype(np.float32)

        ds = TensorDataset(torch.from_numpy(x_s), torch.from_numpy(y_s))
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        self.members = []
        for i in range(self.cfg.n_ensemble):
            _set_seed(self.cfg.seed + 1000 + i)
            model = _MLP(in_dim=x.shape[1], out_dim=y.shape[1], hidden=self.cfg.hidden_dim).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            loss_fn = nn.MSELoss()

            model.train()
            for _ in range(self.cfg.reg_epochs):
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()
            self.members.append(model.eval())

    def predict_with_uncertainty(self, x: np.ndarray, batch_size: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
        if not self.members or self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("Regressor is not fitted.")
        x = np.asarray(x, dtype=np.float32)
        x_s = self.x_scaler.transform(x).astype(np.float32)

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for model in self.members:
                chunks: List[np.ndarray] = []
                for s in range(0, x_s.shape[0], batch_size):
                    xb = torch.from_numpy(x_s[s : s + batch_size]).to(self.device)
                    pb = model(xb).cpu().numpy()
                    chunks.append(pb)
                pred_s = np.concatenate(chunks, axis=0)
                pred = self.y_scaler.inverse_transform(pred_s)
                preds.append(pred)
        stack = np.stack(preds, axis=0)
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        return mean, std


class TorchEnsembleClassifier:
    def __init__(self, cfg: ActiveLearningConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.members: List[_MLP] = []
        self.x_scaler: StandardScaler | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).ravel()
        classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_idx = np.vectorize(class_to_idx.get)(y).astype(np.int64)

        self.classes_ = classes
        self.x_scaler = fit_scaler(x)
        x_s = self.x_scaler.transform(x).astype(np.float32)

        ds = TensorDataset(torch.from_numpy(x_s), torch.from_numpy(y_idx))
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        self.members = []
        for i in range(self.cfg.n_ensemble):
            _set_seed(self.cfg.seed + 2000 + i)
            model = _MLP(in_dim=x.shape[1], out_dim=classes.size, hidden=self.cfg.hidden_dim).to(self.device)
            opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            loss_fn = nn.CrossEntropyLoss()

            model.train()
            for _ in range(self.cfg.cls_epochs):
                for xb, yb in loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
            self.members.append(model.eval())

    def predict_proba(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        if not self.members or self.x_scaler is None:
            raise RuntimeError("Classifier is not fitted.")

        x = np.asarray(x, dtype=np.float32)
        x_s = self.x_scaler.transform(x).astype(np.float32)
        member_probs: List[np.ndarray] = []
        with torch.no_grad():
            for model in self.members:
                chunks: List[np.ndarray] = []
                for s in range(0, x_s.shape[0], batch_size):
                    xb = torch.from_numpy(x_s[s : s + batch_size]).to(self.device)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    chunks.append(probs)
                member_probs.append(np.concatenate(chunks, axis=0))
        return np.mean(np.stack(member_probs, axis=0), axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is None:
            raise RuntimeError("Classifier classes are unavailable.")
        return self.classes_[idx]


@dataclass
class ModelBundle:
    regressor: TorchEnsembleRegressor
    classifier: TorchEnsembleClassifier
    train_split: Dict[str, np.ndarray]


def train_models(
    x: np.ndarray,
    y_reg: np.ndarray,
    y_phase: np.ndarray,
    cfg: ActiveLearningConfig,
    device: torch.device | None = None,
) -> ModelBundle:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = split_train_val(x, y_reg, y_phase, cfg.val_fraction, cfg.seed)
    reg = TorchEnsembleRegressor(cfg, device=device)
    cls = TorchEnsembleClassifier(cfg, device=device)
    reg.fit(split["x_train"], split["y_reg_train"])
    cls.fit(split["x_train"], split["y_phase_train"])
    return ModelBundle(regressor=reg, classifier=cls, train_split=split)


def predict_models(
    bundle: ModelBundle,
    x: np.ndarray,
) -> Dict[str, np.ndarray]:
    reg_mean, reg_std = bundle.regressor.predict_with_uncertainty(x)
    probs = bundle.classifier.predict_proba(x)
    phase_pred = bundle.classifier.predict(x)

    cls_unc = 1.0 - np.max(probs, axis=1)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    entropy = -np.sum(probs * log_probs, axis=1)
    entropy /= np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0

    return {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "phase_pred": phase_pred,
        "phase_proba": probs,
        "cls_uncertainty": cls_unc,
        "cls_entropy": entropy,
    }


def regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    return np.sqrt(mse)


def classification_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

