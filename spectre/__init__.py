"""
OOD signal extractors for SPECTRE-G2.

Eight complementary signals:
  GaussScore : min_c ||z - mu_c||^2  (exact chi-sq test, main novelty)
  FtMahaP    : class-conditional Mahalanobis on PlainNet features
  InMaha     : input-space Mahalanobis
  Energy     : -log sum_c exp(logit_c)
  Entropy    : H[p(y|x)]
  MI         : ensemble mutual information
  ODIN       : temperature-scaled perturbed confidence
  USD        : P(OOD | Gaussian noise binary classifier)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.covariance import LedoitWolf, EmpiricalCovariance


# ── Feature extractors ────────────────────────────────────────────────────────

@torch.no_grad()
def get_gauss_score(models, X: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """Ensemble-averaged GaussScore.  Higher = more OOD."""
    scores = []
    for m in models:
        m.eval(); out = []
        Xt = torch.tensor(X, dtype=torch.float32)
        for i in range(0, len(Xt), batch):
            out.append(m.ood_score(Xt[i:i+batch].to(device)).cpu().numpy())
        scores.append(np.concatenate(out))
    return np.mean(scores, 0).astype(np.float64)


@torch.no_grad()
def get_plain_pen(model, X: np.ndarray, device, batch: int = 512) -> np.ndarray:
    """PlainNet penultimate features."""
    model.eval(); out = []
    Xt = torch.tensor(X, dtype=torch.float32)
    for i in range(0, len(Xt), batch):
        out.append(model.pen(Xt[i:i+batch].to(device)).cpu().numpy())
    return np.vstack(out).astype(np.float64)


@torch.no_grad()
def get_gauss_logits(models, X: np.ndarray, device, batch: int = 512):
    """List of logit arrays (one per ensemble member)."""
    per_member = []
    Xt = torch.tensor(X, dtype=torch.float32)
    for m in models:
        m.eval(); out = []
        for i in range(0, len(Xt), batch):
            out.append(m(Xt[i:i+batch].to(device)).cpu().numpy())
        per_member.append(np.vstack(out).astype(np.float64))
    return per_member


# ── Logit-based signals ───────────────────────────────────────────────────────

def s_energy(logits_list) -> np.ndarray:
    l = np.mean(logits_list, 0)
    return -np.log(np.exp(l - l.max(1, keepdims=True)).sum(1) + 1e-9)


def s_entropy(logits_list) -> np.ndarray:
    l = np.mean(logits_list, 0)
    p = np.exp(l - l.max(1, keepdims=True)); p /= p.sum(1, keepdims=True) + 1e-9
    return -(p * np.log(p + 1e-9)).sum(1)


def s_ens_mi(logits_list) -> np.ndarray:
    probs = []
    for l in logits_list:
        p = np.exp(l - l.max(1, keepdims=True)); p /= p.sum(1, keepdims=True) + 1e-9
        probs.append(p)
    probs = np.stack(probs); mp = probs.mean(0)
    return (-(mp * np.log(mp + 1e-9)).sum(1)
            + (probs * np.log(probs + 1e-9)).sum(-1).mean(0))


# ── ODIN ─────────────────────────────────────────────────────────────────────

def s_odin(models, X: np.ndarray, device,
           T: float = 1000.0, eps: float = 0.001, batch: int = 256) -> np.ndarray:
    """ODIN confidence score.  Higher = more in-distribution."""
    all_sc = []
    for m in models:
        m.eval(); sc = []
        Xt = torch.tensor(X, dtype=torch.float32)
        for i in range(0, len(Xt), batch):
            xb = Xt[i:i+batch].to(device).requires_grad_(True)
            loss = torch.log_softmax(m(xb) / T, 1).max(1).values.mean()
            loss.backward()
            with torch.no_grad():
                conf = torch.softmax(
                    m((xb - eps * xb.grad.sign()).detach()) / T, 1
                ).max(1).values
            sc.append(conf.cpu().detach().numpy()); xb.grad = None
        all_sc.append(np.concatenate(sc))
    return np.mean(all_sc, 0)


# ── Mahalanobis signals ───────────────────────────────────────────────────────

def s_feat_maha_plain(ftr: np.ndarray, ytr: np.ndarray, n_cls: int,
                       *arrays) -> list:
    """Class-conditional Mahalanobis distance on PlainNet features."""
    means = np.array([ftr[ytr == c].mean(0) if (ytr == c).sum() > 0
                       else ftr.mean(0) for c in range(n_cls)])
    resid = np.vstack([ftr[ytr == c] - means[c]
                       for c in range(n_cls) if (ytr == c).sum() > 1])
    try:    prec = LedoitWolf().fit(resid).precision_
    except: prec = np.eye(ftr.shape[1])

    def _s(f):
        return np.stack(
            [np.einsum("ni,ij,nj->n", f - mu, prec, f - mu) for mu in means], 1
        ).min(1)

    return [_s(a) for a in arrays]


def s_input_maha(Xtr: np.ndarray, *arrays) -> list:
    """Global Mahalanobis distance in input space."""
    D = Xtr.shape[1]
    try:
        prec = (EmpiricalCovariance() if D <= 20 else LedoitWolf()).fit(Xtr).precision_
    except:
        prec = np.eye(D)
    mu = Xtr.mean(0)

    def _s(X):
        d = X.astype(np.float64) - mu
        return np.einsum("ni,ij,nj->n", d, prec, d)

    return [_s(a) for a in arrays]


# ── USD ───────────────────────────────────────────────────────────────────────

def s_usd(Xtr: np.ndarray, rng, device,
          n_pseudo: int = 4000, epochs: int = 20, batch: int = 256):
    """
    Train a binary in-dist / Gaussian-noise classifier (USD signal).
    Returns a scoring function.  Higher output = more OOD.
    """
    X_noise = rng.normal(Xtr.mean(0), Xtr.std(0) * 2,
                          (n_pseudo, Xtr.shape[1])).astype(np.float32)
    X_bin = np.vstack([Xtr, X_noise])
    y_bin = np.array([0] * len(Xtr) + [1] * len(X_noise), dtype=np.int64)
    m = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    dl = DataLoader(TensorDataset(torch.tensor(X_bin), torch.tensor(y_bin)),
                    batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            F.cross_entropy(m(xb.to(device)), yb.to(device)).backward()
            opt.step()

    def score(X):
        m.eval()
        out = []
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            for i in range(0, len(Xt), batch):
                out.append(torch.softmax(m(Xt[i:i+batch].to(device)), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    return score


# ── Causal residual ───────────────────────────────────────────────────────────

def s_causal(Xtr: np.ndarray, max_dim: int = 30):
    """
    Causal residual scorer: fit per-variable MLP regressors, use
    standardised residuals as OOD signal.  Only used for D <= max_dim.
    Returns None if D > max_dim.
    """
    if Xtr.shape[1] > max_dim:
        return None
    from sklearn.neural_network import MLPRegressor
    regs, stds = [], []
    for j in range(Xtr.shape[1]):
        reg = MLPRegressor((64, 32), max_iter=300, random_state=42,
                           early_stopping=True, validation_fraction=0.1)
        Xp, yp = np.delete(Xtr, j, 1), Xtr[:, j]
        reg.fit(Xp, yp)
        stds.append(max(float(np.std(yp - reg.predict(Xp))), 1e-6))
        regs.append(reg)

    def score(X):
        t = np.zeros(len(X))
        for j, (reg, std) in enumerate(zip(regs, stds)):
            t += ((X[:, j] - reg.predict(np.delete(X, j, 1))) / std) ** 2
        return t / Xtr.shape[1]

    return score
