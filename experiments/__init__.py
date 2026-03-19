#!/usr/bin/env python
"""
run_benchmark.py — Full benchmark: 12 baselines + SPECTRE-G2 on 4 datasets.

Usage
-----
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --config configs/default.yaml
    python experiments/run_benchmark.py --seed 42 --output results/

Outputs
-------
    results/benchmark_results.csv   — per-split metrics (AUROC, AUPR, FPR95)
    results/benchmark_summary.csv   — dataset-level averages
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import make_synthetic, make_adult, make_cifar10, make_gridworld
from spectre.model import GaussEnc, PlainNet
from spectre.trainer import train_gauss_enc, train_plain
from spectre.signals import (get_gauss_score, get_plain_pen, get_gauss_logits,
                               s_energy, s_entropy, s_ens_mi, s_odin, s_usd,
                               s_feat_maha_plain, s_input_maha, s_causal)
from spectre.combination import topk_combine
from baselines._utils import make_loader, compute_metrics
import baselines


BASELINES = {
    "DeepEnsembles": baselines.run_deep_ensembles,
    "MCDropout":     baselines.run_mc_dropout,
    "BNN":           baselines.run_bnn,
    "BENN":          baselines.run_benn,
    "Evidential":    baselines.run_evidential,
    "DUQ":           baselines.run_duq,
    "Conformal":     baselines.run_conformal,
    "UTraCE":        baselines.run_utrace,
    "CQR":           baselines.run_cqr,
    "ODIN":          baselines.run_odin,
    "Mahalanobis":   baselines.run_mahalanobis,
    "USD":           baselines.run_usd,
}


def run_spectre_g2(Xtr, ytr, Xsc, ytr_full, Xval, yval,
                   te_nn, n_cls, cfg, device, seed):
    rng = np.random.default_rng(seed)
    D   = Xtr.shape[1]
    lam = cfg["spectre"]["lam_gauss_tabular"] if D <= cfg["spectre"]["tabular_dim_threshold"]           else cfg["spectre"]["lam_gauss_highdim"]

    n_ens  = cfg["model"]["n_ensemble"]
    epochs = cfg["training"]["epochs"]
    lr     = cfg["training"]["lr"]
    ltr    = make_loader(Xtr, ytr, cfg["training"]["batch_size"])
    lval   = make_loader(Xval, yval, cfg["training"]["batch_size"], shuffle=False)

    gauss_models = []
    for i in range(n_ens):
        torch.manual_seed(seed + i)
        m = GaussEnc(D, n_cls, cfg["model"]["latent_dim"],
                     cfg["model"]["hidden_dim"], cfg["model"]["dropout_spectral"]).to(device)
        train_gauss_enc(m, ltr, lval, device, epochs, lr, lam_gauss=lam)
        gauss_models.append(m)

    torch.manual_seed(seed + 99)
    plain_m = PlainNet(D, n_cls, cfg["model"]["hidden_dim"],
                       cfg["model"]["dropout_plain"]).to(device)
    train_plain(plain_m, ltr, lval, device, epochs, lr)

    N_p   = cfg["spectre"]["n_pseudo_ood"]
    idx1  = rng.integers(0, len(Xtr), N_p); idx2 = rng.integers(0, len(Xtr), N_p)
    alpha = rng.uniform(cfg["spectre"]["pseudo_ood_alpha_min"],
                        cfg["spectre"]["pseudo_ood_alpha_max"], (N_p,1)).astype(np.float32)
    X_ood = np.vstack([Xtr[idx1]*alpha + Xtr[idx2]*(1-alpha),
                       rng.normal(0, float(Xtr.std())*cfg["spectre"]["pseudo_ood_noise_scale"],
                                   (N_p, D)).astype(np.float32)])

    ftr_P  = get_plain_pen(plain_m, Xsc, device)
    Gv, Go = get_gauss_score(gauss_models, Xval, device), get_gauss_score(gauss_models, X_ood, device)
    fvP, foP = get_plain_pen(plain_m, Xval, device), get_plain_pen(plain_m, X_ood, device)
    BPv, BPo = s_feat_maha_plain(ftr_P, ytr_full, n_cls, fvP, foP)
    Av, Ao   = s_input_maha(Xsc, Xval, X_ood)
    lvl = get_gauss_logits(gauss_models, Xval, device)
    lol = get_gauss_logits(gauss_models, X_ood, device)
    Cv,Co = s_energy(lvl), s_energy(lol)
    Dv,Do = s_entropy(lvl), s_entropy(lol)
    Ev,Eo = s_ens_mi(lvl), s_ens_mi(lol)
    Fv = s_odin(gauss_models, Xval, device); Fo = s_odin(gauss_models, X_ood, device)
    usd_fn = s_usd(Xsc, rng, device, n_pseudo=cfg["spectre"]["usd_n_pseudo"],
                   epochs=cfg["spectre"]["usd_epochs"])
    Hv, Ho = usd_fn(Xval), usd_fn(X_ood)

    val_sigs  = [Gv,BPv,Av,Cv,Dv,Ev,Fv,Hv]
    ood_sigs  = [Go,BPo,Ao,Co,Do,Eo,Fo,Ho]
    snames    = ["GaussScore","FtMahaP","InMaha","Energy","Entropy","MI","ODIN","USD"]

    if D <= cfg["spectre"]["causal_max_dim"]:
        cfn = s_causal(Xsc, max_dim=cfg["spectre"]["causal_max_dim"])
        if cfn:
            val_sigs.append(cfn(Xval)); ood_sigs.append(cfn(X_ood)); snames.append("Causal")
    else:
        cfn = None

    results = {}
    for tname, Xte in te_nn.items():
        Gt   = get_gauss_score(gauss_models, Xte, device)
        fteP = get_plain_pen(plain_m, Xte, device)
        BPt  = s_feat_maha_plain(ftr_P, ytr_full, n_cls, fteP)[0]
        At   = s_input_maha(Xsc, Xte)[0]
        lte  = get_gauss_logits(gauss_models, Xte, device)
        te_sigs = [Gt,BPt,At,s_energy(lte),s_entropy(lte),s_ens_mi(lte),
                   s_odin(gauss_models,Xte,device), usd_fn(Xte)]
        if cfn: te_sigs.append(cfn(Xte))
        s_in, s_te, _, _ = topk_combine(val_sigs, ood_sigs, te_sigs, snames,
                                          top_k=cfg["spectre"]["top_k"],
                                          top_k_threshold=cfg["spectre"]["top_k_threshold"])
        results[tname] = compute_metrics(s_in, s_te)
    return results


def prepare_tabular(splits, is_reg, seed, cfg):
    tr   = splits["train"]
    keys = [k for k in splits if k != "train"]
    tc   = "Y" if is_reg else "target"
    fc   = [c for c in tr.columns if c != tc]
    med  = float(np.median(tr[tc].values)) if is_reg else None
    n_cls = 2 if is_reg else int(tr[tc].nunique())

    def prep(df):
        X = df[[c for c in fc if c in df.columns]].values.astype(np.float32)
        if X.shape[1] < len(fc):
            X = np.hstack([X, np.zeros((len(X), len(fc)-X.shape[1]), np.float32)])
        y = ((df[tc].values > med) if is_reg else df[tc].values).astype(int)             if tc in df.columns else np.zeros(len(df), int)
        return X, y

    Xfull, yfull = prep(tr)
    sc    = StandardScaler(); Xsc = sc.fit_transform(Xfull).astype(np.float32)
    Xtr_, Xval_, ytr_, yval_ = train_test_split(Xsc, yfull, test_size=cfg["evaluation"]["test_size"],
                                                  random_state=seed)
    te_nn = {k: sc.transform(prep(splits[k])[0]).astype(np.float32) for k in keys}
    return Xtr_, ytr_, Xsc, yfull, Xval_, yval_, te_nn, n_cls, keys


def prepare_image(splits, seed, cfg):
    Xfull, yfull = splits["train"]
    keys  = [k for k in splits if k != "train"]
    n_cls = int(yfull.max()) + 1
    sc    = StandardScaler(); Xsc = sc.fit_transform(Xfull).astype(np.float32)
    Xtr_, Xval_, ytr_, yval_ = train_test_split(Xsc, yfull, test_size=cfg["evaluation"]["test_size"],
                                                  random_state=seed)
    te_nn = {k: sc.transform(splits[k][0]).astype(np.float32) for k in keys}
    return Xtr_, ytr_, Xsc, yfull, Xval_, yval_, te_nn, n_cls, keys


def prepare_gridworld(splits, seed, cfg):
    FC   = ["ax","ay","ox","oy","prox"]; TGT = "rew"
    tr   = splits["train"]; keys = [k for k in splits if k != "train"]
    Xf   = tr[FC].values.astype(np.float32); yf = (tr[TGT].values > 0).astype(int)
    sc   = StandardScaler(); Xsc = sc.fit_transform(Xf).astype(np.float32)
    Xtr_, Xval_, ytr_, yval_ = train_test_split(Xsc, yf, test_size=cfg["evaluation"]["test_size"],
                                                  random_state=seed)
    te_nn = {k: sc.transform(splits[k][FC].values.astype(np.float32)) for k in keys}
    return Xtr_, ytr_, Xsc, yf, Xval_, yval_, te_nn, 2, keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed",   type=int, default=None)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed   = args.seed if args.seed is not None else cfg["evaluation"]["seeds"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print(f"Device: {device}  |  Seed: {seed}")
    np.random.seed(seed); torch.manual_seed(seed)

    # Load datasets
    print("\nLoading datasets...")
    datasets = {
        "Synthetic":  (make_synthetic(**cfg["data"]["synthetic"]), "tabular_reg"),
        "Adult":      (make_adult(**cfg["data"]["adult"]),         "tabular_cls"),
        "CIFAR-10":   (make_cifar10(**cfg["data"]["cifar10"]),     "image"),
        "Gridworld":  (make_gridworld(**cfg["data"]["gridworld"]), "gridworld"),
    }

    all_rows = []
    for ds_name, (splits, dtype) in datasets.items():
        print(f"\n{'='*60}\n  {ds_name}\n{'='*60}")

        if dtype == "tabular_reg":
            prep = prepare_tabular(splits, True, seed, cfg)
        elif dtype == "tabular_cls":
            prep = prepare_tabular(splits, False, seed, cfg)
        elif dtype == "image":
            prep = prepare_image(splits, seed, cfg)
        else:
            prep = prepare_gridworld(splits, seed, cfg)

        Xtr_, ytr_, Xsc, yfull, Xval_, yval_, te_nn, n_cls, keys = prep

        # ── Baselines ──────────────────────────────────────────────────────
        for bname, fn in BASELINES.items():
            t0 = time.time()
            print(f"  [{bname}]", end=" ", flush=True)
            try:
                res = fn(Xtr_, ytr_, Xval_, yval_, te_nn, n_cls, device, seed=seed,
                         epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"])
                for tname, m in res.items():
                    all_rows.append(dict(Dataset=ds_name, Model=bname, TestSet=tname, Seed=seed,
                                         AUROC=round(m["auroc"],4), AUPR=round(m["aupr"],4),
                                         FPR95=round(m["fpr95"],4)))
            except Exception as e:
                print(f"ERROR: {e}")
                for tname in keys:
                    all_rows.append(dict(Dataset=ds_name, Model=bname, TestSet=tname, Seed=seed,
                                         AUROC=0.5, AUPR=0.5, FPR95=1.0))
            print(f"{time.time()-t0:.1f}s")

        # ── SPECTRE-G2 ────────────────────────────────────────────────────
        t0 = time.time(); print("  [SPECTRE-G2]", end=" ", flush=True)
        try:
            res = run_spectre_g2(Xtr_, ytr_, Xsc, yfull, Xval_, yval_, te_nn, n_cls, cfg, device, seed)
            for tname, m in res.items():
                all_rows.append(dict(Dataset=ds_name, Model="SPECTRE_G2", TestSet=tname, Seed=seed,
                                     AUROC=round(m["auroc"],4), AUPR=round(m["aupr"],4),
                                     FPR95=round(m["fpr95"],4)))
        except Exception as e:
            print(f"ERROR: {e}")
            for tname in keys:
                all_rows.append(dict(Dataset=ds_name, Model="SPECTRE_G2", TestSet=tname, Seed=seed,
                                     AUROC=0.5, AUPR=0.5, FPR95=1.0))
        print(f"{time.time()-t0:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.output, "benchmark_results.csv"), index=False)

    summary = (df[df.TestSet != "regular"]
               .groupby(["Dataset","Model"])[["AUROC","AUPR","FPR95"]]
               .mean().round(4).reset_index())
    summary.to_csv(os.path.join(args.output, "benchmark_summary.csv"), index=False)

    print(f"\n✓ Saved to {args.output}/")
    pivot = summary.pivot_table(index="Dataset", columns="Model", values="AUROC").round(4)
    print("\nMean AUROC across anomaly splits:")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
