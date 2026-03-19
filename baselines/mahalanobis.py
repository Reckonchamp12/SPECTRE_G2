"""UTraCE simplified — entropy-based OOD detector."""

import numpy as np
import torch
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_utrace(Xtr, ytr, Xval, yval, Xte_sets: dict,
               n_cls: int, device, seed: int = 42,
               epochs: int = 30, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    def entropy(X):
        p = get_probs(m, X, device)
        return -(p * np.log(p + 1e-9)).sum(1)
    H_in = entropy(Xval)
    results = {}
    for name, Xte in Xte_sets.items():
        results[name] = compute_metrics(H_in, entropy(Xte))
    return results
