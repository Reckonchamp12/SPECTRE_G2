"""BENN: Bayesian Entropy Neural Network."""

import numpy as np
import torch
import torch.nn as nn
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, get_probs, compute_metrics


def run_benn(Xtr, ytr, Xval, yval, Xte_sets: dict,
             n_cls: int, device, seed: int = 42,
             epochs: int = 30, lr: float = 1e-3, n_mc: int = 30) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls, dropout=0.2).to(device)
    ce = nn.CrossEntropyLoss()
    def loss_fn(logits, y):
        p = torch.softmax(logits, 1)
        H = -torch.sum(p * torch.log(p + 1e-9), 1).mean()
        return ce(logits, y) - 0.01 * H
    train_model(m, ltr, epochs=epochs, lr=lr, loss_fn=loss_fn, val_loader=lval, device=device)
    m.train()
    def mc_probs(X):
        ps = [get_probs(m, X, device, mc=True, n_mc=1)[0] for _ in range(n_mc)]
        return np.stack(ps).mean(0)
    conf_in = mc_probs(Xval).max(1)
    results = {}
    for name, Xte in Xte_sets.items():
        conf_out = mc_probs(Xte).max(1)
        results[name] = compute_metrics(-conf_in, -conf_out)
    return results
