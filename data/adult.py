"""
Gridworld RL dataset — agent navigation with reward-based anomaly types.

Anomaly types
-------------
mechanism : step-function reward replaces continuous linear reward
newobj    : third object type (C) with novel reward structure added
"""

import math
import numpy as np
import pandas as pd


def make_gridworld(n_train: int = 5_000, n_test: int = 1_000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    G    = 10
    ACTS = [(0,1),(0,-1),(1,0),(-1,0)]

    def prox(ax,ay,ox,oy):
        return 1.0 / (1.0 + math.sqrt((ax-ox)**2 + (ay-oy)**2))

    def rew_base(p,t):
        return (1.5*p if t=="A" else -1.0*p)

    def rew_step(p,t):
        if t=="A": return 2.0 if p > 0.5 else 0.1
        return -2.0 if p > 0.5 else -0.1

    def rew_new(p,t):
        if t=="C": return 2.0 if 0.3 < p <= 0.7 else -0.5
        return rew_base(p,t)

    def rollout(n,types,rfn):
        ax,ay = rng.integers(0,G,2).tolist()
        ox,oy = rng.integers(0,G,2).tolist()
        ot    = rng.choice(types); rows = []
        for s in range(n):
            a = rng.integers(0,4); da,db = ACTS[a]
            ax = int(np.clip(ax+da, 0, G-1))
            ay = int(np.clip(ay+db, 0, G-1))
            p  = prox(ax,ay,ox,oy); r = rfn(p,ot)
            rows.append([ax,ay,ox,oy,p,a,r,ot])
            if s % 200 == 199:
                ox,oy = rng.integers(0,G,2).tolist()
                ot    = rng.choice(types)
        return pd.DataFrame(rows, columns=["ax","ay","ox","oy","prox","act","rew","type"])

    return dict(
        train     = rollout(n_train, ["A","B"], rew_base),
        regular   = rollout(n_test,  ["A","B"], rew_base),
        newobj    = rollout(n_test,  ["A","B","C"], rew_new),
        mechanism = rollout(n_test,  ["A","B"], rew_step),
    )
