#!/usr/bin/env python
"""
run_multiseed.py — Multi-seed reproducibility sweep.

Usage
-----
    python experiments/run_multiseed.py
    python experiments/run_multiseed.py --seeds 42 123 777 --output results/

Outputs
-------
    results/multiseed_raw.csv        — every (seed, dataset, model, split) row
    results/multiseed_mean.csv       — mean AUROC pivot table
    results/multiseed_std.csv        — std AUROC pivot table
    results/multiseed_meanstd.csv    — "0.742 +/- 0.008" formatted strings
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_benchmark import main as benchmark_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_dfs = []
    for seed in args.seeds:
        print(f"\n{'#'*60}\n  SEED = {seed}\n{'#'*60}")
        sys.argv = ["run_benchmark.py", "--config", args.config,
                    "--seed", str(seed), "--output", args.output + f"/seed_{seed}"]
        os.makedirs(args.output + f"/seed_{seed}", exist_ok=True)
        benchmark_main()
        df = pd.read_csv(f"{args.output}/seed_{seed}/benchmark_results.csv")
        all_dfs.append(df)

    raw = pd.concat(all_dfs, ignore_index=True)
    raw.to_csv(f"{args.output}/multiseed_raw.csv", index=False)

    agg = (raw[raw.TestSet != "regular"]
           .groupby(["Dataset","Model","TestSet"])[["AUROC","AUPR","FPR95"]]
           .agg(["mean","std"]).round(4).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]

    ORDERED = ["DeepEnsembles","MCDropout","BNN","BENN","Evidential","DUQ",
               "Conformal","UTraCE","CQR","ODIN","Mahalanobis","USD","SPECTRE_G2"]

    for metric in ["AUROC","AUPR","FPR95"]:
        pm = (agg.pivot_table(index=["Dataset","TestSet"], columns="Model",
                               values=f"{metric}_mean").round(4))
        ps = (agg.pivot_table(index=["Dataset","TestSet"], columns="Model",
                               values=f"{metric}_std").round(4))
        pm = pm[[c for c in ORDERED if c in pm.columns]]
        ps = ps[[c for c in ORDERED if c in ps.columns]]
        pm.to_csv(f"{args.output}/multiseed_{metric.lower()}_mean.csv")
        ps.to_csv(f"{args.output}/multiseed_{metric.lower()}_std.csv")

        ms = pd.DataFrame(index=pm.index, columns=pm.columns, dtype=str)
        for col in pm.columns:
            for idx in pm.index:
                mu  = pm.loc[idx, col]; sd = ps.loc[idx, col]
                ms.loc[idx, col] = f"{mu:.3f}" if pd.isna(sd) else f"{mu:.3f}+/-{sd:.3f}"
        ms.to_csv(f"{args.output}/multiseed_{metric.lower()}_meanstd.csv")
        print(f"\n{metric} (mean +/- std):")
        print(ms.to_string())

    print(f"\n✓ All results saved to {args.output}/")


if __name__ == "__main__":
    main()
