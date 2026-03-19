"""BNN via last-layer Laplace approximation (Daxberger et al., 2021)."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, penultimate, compute_metrics


def run_bnn(Xtr, ytr, Xval, yval, Xte_sets: dict,
            n_cls: int, device, seed: int = 42,
            epochs: int = 30, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    Ftr = penultimate(m, Xtr, device)
    lr_model = LogisticRegression(max_iter=200, C=1.0)
    lr_model.fit(Ftr, ytr)
    conf_in = lr_model.predict_proba(penultimate(m, Xval, device)).max(1)
    results = {}
    for name, Xte in Xte_sets.items():
        conf_out = lr_model.predict_proba(penultimate(m, Xte, device)).max(1)
        results[name] = compute_metrics(-conf_in, -conf_out)
    return results
