"""Standard Conformal Prediction (Vovk et al.) used as OOD detector."""

import numpy as np
import torch
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_conformal(Xtr, ytr, Xval, yval, Xte_sets: dict,
                  n_cls: int, device, seed: int = 42,
                  epochs: int = 30, lr: float = 1e-3, alpha: float = 0.1) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    pr_val = get_probs(m, Xval, device)
    q      = np.quantile(1 - pr_val[np.arange(len(yval)), yval], 1 - alpha)
    def set_size(X):
        return (get_probs(m, X, device) >= (1 - q)).sum(1).astype(float)
    sz_in = set_size(Xval)
    results = {}
    for name, Xte in Xte_sets.items():
        results[name] = compute_metrics(-sz_in, -set_size(Xte))
    return results
