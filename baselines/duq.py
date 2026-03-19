"""Mahalanobis distance OOD detector (Lee et al., 2018)."""

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, penultimate, compute_metrics


def run_mahalanobis(Xtr, ytr, Xval, yval, Xte_sets: dict,
                    n_cls: int, device, seed: int = 42,
                    epochs: int = 30, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    Ftr   = penultimate(m, Xtr, device)
    means = [Ftr[ytr == c].mean(0) if (ytr == c).sum() > 1 else Ftr.mean(0)
             for c in range(n_cls)]
    prec  = EmpiricalCovariance().fit(Ftr).precision_

    def score(X):
        F = penultimate(m, X, device)
        dists = [np.einsum("ni,ij,nj->n", F - mu, prec, F - mu) for mu in means]
        return -np.stack(dists, 1).min(1)

    sc_in = score(Xval)
    results = {}
    for name, Xte in Xte_sets.items():
        results[name] = compute_metrics(-sc_in, -score(Xte))
    return results
