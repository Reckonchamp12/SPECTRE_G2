"""CQR / Adaptive Prediction Sets (Romano et al., 2020)."""

import numpy as np
import torch
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_cqr(Xtr, ytr, Xval, yval, Xte_sets: dict,
            n_cls: int, device, seed: int = 42,
            epochs: int = 30, lr: float = 1e-3, quantile: float = 0.9) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)

    def aps(probs, labels=None):
        order = np.argsort(-probs, 1); scores = []
        for i, row in enumerate(probs):
            cum = np.cumsum(row[order[i]])
            if labels is not None:
                rank = np.where(order[i] == labels[i])[0][0]
                scores.append(cum[rank])
            else:
                scores.append(cum)
        return np.array(scores) if labels is not None else scores

    q = np.quantile(aps(get_probs(m, Xval, device), yval), quantile)
    def set_size(X):
        pr = get_probs(m, X, device)
        return np.array([np.searchsorted(np.cumsum(np.sort(pr[i])[::-1]), q) + 1
                         for i in range(len(pr))])
    sz_in = set_size(Xval).astype(float)
    results = {}
    for name, Xte in Xte_sets.items():
        results[name] = compute_metrics(-sz_in, -set_size(Xte).astype(float))
    return results
