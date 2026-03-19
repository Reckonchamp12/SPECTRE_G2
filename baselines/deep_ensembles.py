"""MC Dropout baseline (Gal & Ghahramani, 2016)."""

import numpy as np
import torch
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_mc_dropout(Xtr, ytr, Xval, yval, Xte_sets: dict,
                   n_cls: int, device, seed: int = 42,
                   epochs: int = 30, lr: float = 1e-3, n_mc: int = 30) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls, dropout=0.3).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr, val_loader=lval, device=device)
    pr_in = get_probs(m, Xval, device, mc=True, n_mc=n_mc).mean(0)
    conf_in = pr_in.max(1)
    results = {}
    for name, Xte in Xte_sets.items():
        pr_out   = get_probs(m, Xte, device, mc=True, n_mc=n_mc).mean(0)
        conf_out = pr_out.max(1)
        results[name] = compute_metrics(-conf_in, -conf_out)
    return results
