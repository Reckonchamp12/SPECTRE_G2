"""
Synthetic dataset — 5-variable causal SCM with four anomaly types.

DAG structure:
    X1 → X2 → X4 → Y
    X1 → X3 → X5 → Y
    X1 → Y

Anomaly types
-------------
confounder  : hidden variable U affects X2 and X4 simultaneously
newvar      : a new causal variable X6 is introduced (X6 → Y)
mechanism   : the X2 → X4 mechanism changes from linear to quadratic
interaction : a multiplicative interaction X2 * X3 is added to Y
"""

import numpy as np
import pandas as pd


def make_synthetic(n_train: int = 10_000, n_test: int = 2_000, seed: int = 42) -> dict:
    """
    Generate the synthetic causal SCM dataset.

    Returns
    -------
    dict with keys: 'train', 'regular', 'confounder', 'newvar', 'mechanism', 'interaction'
    Each value is a pd.DataFrame with columns X1..X5 (X6 for newvar), Y.
    """
    rng = np.random.default_rng(seed)

    def noise(n: int) -> np.ndarray:
        return rng.normal(0, 0.3, n)

    def base(n: int) -> pd.DataFrame:
        X1 = rng.normal(0, 1, n)
        X2 = 0.8 * X1 + noise(n)
        X3 = -0.5 * X1 + 0.4 * X1 ** 2 + noise(n)
        X4 = 0.7 * X2 + noise(n)
        X5 = np.tanh(0.9 * X3) + noise(n)
        Y  = 0.6 * X4 + 0.5 * X5 + 0.3 * X1 + noise(n)
        return pd.DataFrame(dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, Y=Y))

    def confounder(n: int) -> pd.DataFrame:
        """Hidden variable U confounds X2 and X4."""
        U  = rng.normal(0, 1, n)
        X1 = rng.normal(0, 1, n)
        X2 = 0.8 * X1 + 0.6 * U + noise(n)
        X3 = -0.5 * X1 + 0.4 * X1 ** 2 + noise(n)
        X4 = 0.7 * X2 + 0.6 * U + noise(n)
        X5 = np.tanh(0.9 * X3) + noise(n)
        Y  = 0.6 * X4 + 0.5 * X5 + 0.3 * X1 + noise(n)
        return pd.DataFrame(dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, Y=Y))

    def newvar(n: int) -> pd.DataFrame:
        """New variable X6 added with causal effect on Y."""
        df = base(n)
        X6 = rng.normal(0, 1, n)
        df["X6"] = X6
        df["Y"]  = df["Y"] + 0.8 * X6
        return df

    def mechanism(n: int) -> pd.DataFrame:
        """X2 → X4 mechanism changes from linear to quadratic."""
        X1 = rng.normal(0, 1, n)
        X2 = 0.8 * X1 + noise(n)
        X3 = -0.5 * X1 + 0.4 * X1 ** 2 + noise(n)
        X4 = 0.35 * X2 ** 2 + noise(n)           # was 0.7 * X2
        X5 = np.tanh(0.9 * X3) + noise(n)
        Y  = 0.6 * X4 + 0.5 * X5 + 0.3 * X1 + noise(n)
        return pd.DataFrame(dict(X1=X1, X2=X2, X3=X3, X4=X4, X5=X5, Y=Y))

    def interaction(n: int) -> pd.DataFrame:
        """Multiplicative interaction X2 * X3 added to Y."""
        df = base(n)
        df["Y"] = df["Y"] + 0.5 * df["X2"] * df["X3"]
        return df

    return dict(
        train       = base(n_train),
        regular     = base(n_test),
        confounder  = confounder(n_test),
        newvar      = newvar(n_test),
        mechanism   = mechanism(n_test),
        interaction = interaction(n_test),
    )
