"""USD: Unknown Sensitive Detector (binary in-dist vs Gaussian noise)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_usd(Xtr, ytr, Xval, yval, Xte_sets: dict,
            n_cls: int, device, seed: int = 42,
            epochs: int = 30, lr: float = 1e-3,
            noise_scale: float = 2.0) -> dict:
    torch.manual_seed(seed)
    rng     = np.random.default_rng(seed)
    X_noise = rng.normal(Xtr.mean(0), Xtr.std(0) * noise_scale,
                          Xtr.shape).astype(np.float32)
    X_bin   = np.vstack([Xtr, X_noise])
    y_bin   = np.array([0] * len(Xtr) + [1] * len(X_noise))
    ltr_bin = make_loader(X_bin, y_bin)
    lval_bin = make_loader(
        np.vstack([Xval, X_noise[:len(Xval)]]),
        np.array([0] * len(Xval) + [1] * len(Xval[:len(Xval)])),
        shuffle=False)
    m = MLP(Xtr.shape[1], 2).to(device)
    from spectre.trainer import train_model
    train_model(m, ltr_bin, epochs=epochs, lr=lr, val_loader=lval_bin, device=device)
    sc_in = get_probs(m, Xval, device)[:, 1]
    results = {}
    for name, Xte in Xte_sets.items():
        sc_out = get_probs(m, Xte, device)[:, 1]
        results[name] = compute_metrics(sc_in, sc_out)
    return results
