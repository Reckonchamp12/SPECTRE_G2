"""ODIN: Out-of-DIstribution detector for Neural networks (Liang et al., 2018)."""

import numpy as np
import torch
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, compute_metrics


def run_odin(Xtr, ytr, Xval, yval, Xte_sets: dict,
             n_cls: int, device, seed: int = 42,
             epochs: int = 30, lr: float = 1e-3,
             T: float = 1000.0, eps: float = 0.002) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    m.eval()

    def score(X):
        Xt = torch.tensor(X, dtype=torch.float32).to(device).requires_grad_(True)
        logits = m(Xt)
        loss   = -torch.log_softmax(logits / T, 1).max(1).values.mean()
        loss.backward()
        with torch.no_grad():
            Xt_p = Xt - eps * Xt.grad.sign()
            p    = torch.softmax(m(Xt_p) / T, 1)
        return p.max(1).values.cpu().numpy()

    sc_in = score(Xval)
    results = {}
    for name, Xte in Xte_sets.items():
        results[name] = compute_metrics(-sc_in, -score(Xte))
    return results
