"""Training loops for GaussEnc, PlainNet, and the generic MLP."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_gauss_enc(model, ltr: DataLoader, lval: DataLoader,
                    device, epochs: int, lr: float,
                    lam_gauss: float = 0.5) -> None:
    """Train GaussEnc jointly with classification + Gaussianization loss."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, wait, bw = 1e9, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            L = F.cross_entropy(model(xb), yb) + lam_gauss * model.gaussianization_loss(xb, yb)
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval(); vl = nb = 0
        with torch.no_grad():
            for xb, yb in lval:
                vl += F.cross_entropy(model(xb.to(device)), yb.to(device)).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best - 1e-5: best, wait, bw = vl, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 8: break
    if bw: model.load_state_dict(bw)


def train_plain(model, ltr: DataLoader, lval: DataLoader,
                device, epochs: int, lr: float) -> None:
    """Train PlainNet with standard cross-entropy."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, wait, bw = 1e9, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in ltr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval(); vl = nb = 0
        with torch.no_grad():
            for xb, yb in lval:
                vl += F.cross_entropy(model(xb.to(device)), yb.to(device)).item(); nb += 1
        vl /= max(nb, 1)
        if vl < best - 1e-5: best, wait, bw = vl, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 8: break
    if bw: model.load_state_dict(bw)


def train_model(model, loader, epochs: int = 30, lr: float = 1e-3,
                wd: float = 1e-4, loss_fn=None, val_loader=None,
                patience: int = 5, device=None) -> None:
    """Generic training loop for CEN baseline MLPs."""
    if device is None:
        device = next(model.parameters()).device
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.to(device); best, wait, bw = 1e9, 0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        if val_loader is not None:
            model.eval(); vl = nb = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    vl += loss_fn(model(xb.to(device)), yb.to(device)).item(); nb += 1
            vl /= nb
            if vl < best: best, wait, bw = vl, 0, copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience: break
    if bw: model.load_state_dict(bw)
