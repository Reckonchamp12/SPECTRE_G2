"""Evidential Deep Learning (Sensoy et al., 2018)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectre.trainer import train_model
from spectre.model import MLP
from ._utils import make_loader, compute_metrics


class _EvidentialLoss(nn.Module):
    def __init__(self, n_cls: int, lam: float = 0.001):
        super().__init__(); self.K = n_cls; self.lam = lam

    def forward(self, logits, y):
        ev = F.softplus(logits); alpha = ev + 1; S = alpha.sum(1, keepdim=True)
        y_oh = F.one_hot(y, self.K).float()
        err  = torch.sum(y_oh * (torch.digamma(S) - torch.digamma(alpha)), 1).mean()
        a_t  = y_oh + (1 - y_oh) * alpha; S_t = a_t.sum(1, keepdim=True)
        K_t  = torch.tensor(float(self.K), device=logits.device)
        kl   = (torch.lgamma(S_t) - torch.lgamma(K_t)
                - torch.lgamma(a_t).sum(1, keepdim=True)
                + (a_t - 1) * (torch.digamma(a_t) - torch.digamma(S_t))
               ).sum(1).mean()
        return err + self.lam * kl


def run_evidential(Xtr, ytr, Xval, yval, Xte_sets: dict,
                   n_cls: int, device, seed: int = 42,
                   epochs: int = 30, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    ltr  = make_loader(Xtr, ytr)
    lval = make_loader(Xval, yval, shuffle=False)
    m = MLP(Xtr.shape[1], n_cls).to(device)
    train_model(m, ltr, epochs=epochs, lr=lr,
                loss_fn=_EvidentialLoss(n_cls), val_loader=lval, device=device)

    def uncertainty(X):
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            ev = F.softplus(m(Xt)); alpha = ev + 1; S = alpha.sum(1)
            return (n_cls / S).cpu().numpy()

    u_in = uncertainty(Xval)
    results = {}
    for name, Xte in Xte_sets.items():
        u_out = uncertainty(Xte)
        results[name] = compute_metrics(-u_in, -u_out)
    return results
