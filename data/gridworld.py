"""
CIFAR-10 dataset — 10-class image classification with pixel-level anomaly types.
Uses a pretrained ResNet-18 to extract 512-d features.

Anomaly types
-------------
confounder  : selective red channel boost on odd-class images
mechanism   : brightness inversion (pixel negation)
newvar      : Gaussian blur corruption
"""

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T
    HAS_TV = True
except ImportError:
    HAS_TV = False


def make_cifar10(n_train: int = 5_000, n_test: int = 1_000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    if not HAS_TV:
        D = 512
        def rand_split(n, n_cls=10):
            X = rng.normal(0,1,(n,D)).astype(np.float32)
            y = rng.integers(0,n_cls,n)
            return X, y
        Xtr, ytr = rand_split(n_train)
        Xte, yte = rand_split(n_test)
        def perturb(X, scale):
            return (X + rng.normal(0,scale,X.shape)).astype(np.float32)
        return dict(
            train=(Xtr,ytr), regular=(Xte,yte),
            mechanism=(perturb(Xte,1.5),yte),
            newvar=(perturb(Xte,0.8),yte),
            confounder=(perturb(Xte,1.0),yte))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
    ds_tr = torchvision.datasets.CIFAR10("./data",train=True, download=True,transform=tf)
    ds_te = torchvision.datasets.CIFAR10("./data",train=False,download=True,transform=tf)
    idx_tr = rng.choice(len(ds_tr), n_train, replace=False)
    idx_te = rng.choice(len(ds_te), n_test,  replace=False)

    def extract(ds, idx):
        Xs, ys = [], []
        for i in idx: x,y = ds[int(i)]; Xs.append(x.numpy()); ys.append(y)
        return np.stack(Xs), np.array(ys)

    Xtr,ytr = extract(ds_tr, idx_tr)
    Xte,yte = extract(ds_te, idx_te)

    model = torchvision.models.resnet18(weights="DEFAULT")
    model.fc = nn.Identity(); model.eval().to(DEVICE)
    def featurize(X):
        feats = []
        with torch.no_grad():
            for i in range(0,len(X),64):
                b = torch.tensor(X[i:i+64]).to(DEVICE)
                feats.append(model(b).cpu().numpy())
        return np.concatenate(feats)

    Ftr = featurize(Xtr); Fte = featurize(Xte)

    def invert_brightness(X):
        return np.clip(X*-1, X.min(), X.max()).astype(np.float32)

    def add_blur(X):
        from PIL import Image, ImageFilter
        out = []
        for img in X:
            pil = Image.fromarray(
                ((img.transpose(1,2,0)*np.array([0.247,0.243,0.261])+
                  np.array([0.4914,0.4822,0.4465]))*255).clip(0,255).astype(np.uint8))
            pil = pil.filter(ImageFilter.GaussianBlur(1.5))
            out.append(tf(pil).numpy())
        return np.stack(out)

    def color_cast(X, mask):
        out = X.copy(); out[mask,0] += 0.5
        return out.astype(np.float32)

    odd_mask = (yte % 2 == 1)
    Fte_mech = featurize(invert_brightness(Xte))
    Fte_newv = featurize(add_blur(Xte))
    Fte_conf = featurize(color_cast(Xte, odd_mask))

    return dict(train=(Ftr,ytr), regular=(Fte,yte),
                mechanism=(Fte_mech,yte), newvar=(Fte_newv,yte), confounder=(Fte_conf,yte))
