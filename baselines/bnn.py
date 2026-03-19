"""Shared utilities for all CEN baselines."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch: int = 256, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def get_probs(model, X: np.ndarray, device, mc: bool = False,
              n_mc: int = 30) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    if mc:
        model.train()
        ps = []
        with torch.no_grad():
            for _ in range(n_mc):
                ps.append(torch.softmax(model(Xt), 1).cpu().numpy())
        return np.stack(ps)   # (n_mc, N, C)
    with torch.no_grad():
        return torch.softmax(model(Xt), 1).cpu().numpy()


def penultimate(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model.features(Xt).cpu().numpy()


def fpr_at_tpr(y_true, scores, tpr_target: float = 0.95) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    idxs = np.where(tpr >= tpr_target)[0]
    return float(fpr[idxs[0]]) if len(idxs) > 0 else 1.0


def compute_metrics(scores_in: np.ndarray, scores_out: np.ndarray) -> dict:
    """Compute AUROC, AUPR, and FPR95."""
    for arr in (scores_in, scores_out):
        finite = arr[np.isfinite(arr)]
        fill   = float(finite.mean()) if len(finite) > 0 else 0.0
        arr[~np.isfinite(arr)] = fill

    labels = np.array([0] * len(scores_in) + [1] * len(scores_out))
    scores = np.concatenate([scores_in, scores_out])

    if len(np.unique(labels)) < 2 or np.allclose(scores, scores[0]):
        return dict(auroc=0.5, aupr=0.5, fpr95=1.0)

    try:
        auroc = float(np.clip(roc_auc_score(labels, scores), 0, 1))
        aupr  = float(np.clip(average_precision_score(labels, scores), 0, 1))
        fpr95 = float(np.clip(fpr_at_tpr(labels, scores), 0, 1))
    except Exception:
        return dict(auroc=0.5, aupr=0.5, fpr95=1.0)

    return dict(auroc=auroc, aupr=aupr, fpr95=fpr95)
