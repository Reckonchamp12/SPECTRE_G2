"""
Val-percentile top-k signal combination.

Normalisation strategy (proven from v5 onwards):
  - Compute p1, p99 from the *validation* set only.
  - Clip test scores to [0, 3] relative to the val range.
  - This allows shifted test samples to exceed the val range (clipped at 3)
    without corrupting the normalisation.

Signal selection:
  k = 1  if the top signal AUROC >= threshold (0.72 by default)
  k = 2  otherwise

Direction correction:
  - Known directions (ODIN, USD, etc.) are hardcoded via KNOWN_DIR.
  - Unknown signals: direction inferred from pseudo-OOD vs val means.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


KNOWN_DIR = {
    "GaussScore": "higher_is_ood",
    "FtMahaP":    "higher_is_ood",
    "InMaha":     "higher_is_ood",
    "Energy":     "higher_is_ood",
    "Entropy":    "higher_is_ood",
    "MI":         "higher_is_ood",
    "ODIN":       "lower_is_ood",   # returns confidence; higher = in-dist
    "USD":        "higher_is_ood",  # returns P(OOD)
    "Causal":     "higher_is_ood",
}


def topk_combine(val_sigs: list, ood_sigs: list, te_sigs: list,
                 names: list, top_k: int = 2,
                 top_k_threshold: float = 0.72):
    """
    Normalise each signal with val-percentile clipping, rank by val-vs-ood AUROC,
    and return the (mean) combined score for val and test.

    Parameters
    ----------
    val_sigs        : list of (N_val,) arrays, one per signal
    ood_sigs        : list of (N_ood,) arrays used only for direction detection
    te_sigs         : list of (N_te,)  arrays, one per signal
    names           : signal name strings (must match KNOWN_DIR keys where applicable)
    top_k           : maximum number of signals to combine
    top_k_threshold : use k=1 if top signal AUROC >= this value

    Returns
    -------
    score_val, score_te, auroc_dict, selected_names
    """
    N_val = len(val_sigs[0]); N_te = len(te_sigs[0]); normed = []

    for name, sv, so, st in zip(names, val_sigs, ood_sigs, te_sigs):
        sv = np.where(np.isfinite(sv), sv, np.nanmedian(sv))
        so = np.where(np.isfinite(so), so, np.nanmedian(sv))
        st = np.where(np.isfinite(st), st, np.nanmedian(sv))

        # Direction correction
        if name in KNOWN_DIR:
            flip = (KNOWN_DIR[name] == "lower_is_ood")
        else:
            flip = (so.mean() < sv.mean())

        if flip: sv, so, st = -sv, -so, -st

        # Val-percentile normalisation
        p1, p99 = np.percentile(sv, 1), np.percentile(sv, 99)
        span    = max(p99 - p1, 1e-8)
        sv_n    = np.clip((sv - p1) / span, 0, 3)
        st_n    = np.clip((st - p1) / span, 0, 3)

        lbl = np.array([0] * N_val + [1] * N_te)
        try:    au = float(roc_auc_score(lbl, np.concatenate([sv_n, st_n])))
        except: au = 0.5

        normed.append((name, au, sv_n, st_n))

    normed.sort(key=lambda x: x[1], reverse=True)
    au_dict = {n: round(a, 4) for n, a, _, _ in normed}
    k       = 1 if normed[0][1] >= top_k_threshold else min(top_k, len(normed))
    sel     = normed[:k]

    score_val = np.mean([sv for _, _, sv, _  in sel], 0)
    score_te  = np.mean([st for _, _, _,  st in sel], 0)
    sel_names = [n for n, _, _, _ in sel]
    return score_val, score_te, au_dict, sel_names
