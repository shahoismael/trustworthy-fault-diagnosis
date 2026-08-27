"""STEP 19 - Open-set baselines vs feature-space Mahalanobis.

Run from anywhere in the repo:
    python final-submission/step19_openset_baselines.py
(needs torch + the step02 processed cache in data/tep_extended/processed/)

Adds literature-standard open-set scores on the SAME unified CNN features and
reports AUROC against the held-out unknown faults (16,17,18), so the paper can
compare its Mahalanobis rejector to competing methods, not just to softmax.

Scores (higher = more 'unknown'):
  MSP, Energy, Entropy, KNN (k=5 feature distance), Conformal (p-value),
  Mahalanobis (ours, Lee et al. 2018).

Output: results/openset_baselines.csv
"""
import sys, os
from pathlib import Path
# make src/config.py importable regardless of where this file sits
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from config import DATA, RESULTS, UNKNOWN_FAULTS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

PROC = DATA / "tep_extended" / "processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return (g("X_train.npy"), g("y_train.npy"), g("X_val.npy"), g("y_val.npy"),
            g("X_test.npy"), g("y_test.npy"))

class CNN(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3))
        self.head = nn.Linear(128, c)
    def features(self, x): return self.body(x.transpose(1, 2))
    def forward(self, x):  return self.head(self.features(x))

def focal(logits, target, gamma=2.0, w=None):
    ce = F.cross_entropy(logits, target, weight=w, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xval, yval, Xte, yte = load()
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
    kcl = np.unique(ytr[known]); kmap = {c: i for i, c in enumerate(kcl)}
    enc = lambda y: np.array([kmap[v] for v in y]); C = len(kcl)
    cnt = np.bincount(enc(ytr[known]), minlength=C).astype(float)
    w = torch.tensor(np.sqrt(cnt.sum()/(C*np.clip(cnt,1,None))), dtype=torch.float32, device=DEV)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr[known]), torch.tensor(enc(ytr[known]), dtype=torch.long)),
                    batch_size=BS, shuffle=True)
    net = CNN(Xtr.shape[-1], C).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); focal(net(xb), yb, w=w).backward(); opt.step()
    net.eval()
    def batch(fn, X, bs=512):
        return torch.cat([fn(torch.tensor(X[i:i+bs]).to(DEV)) for i in range(0, len(X), bs)])
    with torch.no_grad():
        logit_te  = batch(lambda t: net(t).cpu(), Xte)
        feat_te   = batch(lambda t: net.features(t).cpu(), Xte).numpy()
        feat_tr   = batch(lambda t: net.features(t).cpu(), Xtr[known]).numpy()
        logit_val = batch(lambda t: net(t).cpu(), Xval)
    P = F.softmax(logit_te, 1).numpy()
    is_unknown = np.isin(yte, UNKNOWN_FAULTS).astype(int)

    msp    = -P.max(1)
    energy = (-torch.logsumexp(logit_te, 1)).numpy()
    ent    = -(np.clip(P,1e-12,1)*np.log(np.clip(P,1e-12,1))).sum(1)

    ytr_k = enc(ytr[known])
    mus = np.stack([feat_tr[ytr_k==i].mean(0) for i in range(C)])
    cov = np.cov((feat_tr - mus[ytr_k]).T) + 1e-6*np.eye(feat_tr.shape[1])
    prec = np.linalg.pinv(cov)
    d = feat_te[:,None,:]-mus[None]
    maha = np.einsum("ncd,de,nce->nc", d, prec, d).min(1)

    nn_ = NearestNeighbors(n_neighbors=5).fit(feat_tr)
    knn = nn_.kneighbors(feat_te, return_distance=True)[0].mean(1)

    Pval = F.softmax(logit_val, 1).numpy()
    calib = 1.0 - Pval[~np.isin(yval, UNKNOWN_FAULTS)].max(1)
    test_nc = 1.0 - P.max(1)
    pval = (1.0 + (calib[None,:] >= test_nc[:,None]).sum(1)) / (len(calib)+1.0)
    conf = 1.0 - pval

    A = lambda s: round(roc_auc_score(is_unknown, s), 4)
    return {"seed":seed,"MSP":A(msp),"Energy":A(energy),"Entropy":A(ent),
            "KNN":A(knn),"Conformal":A(conf),"Mahalanobis":A(maha)}

def main():
    rows=[run(s) for s in [42,7,123,2024,2025]]
    for r in rows: print(r)
    df=pd.DataFrame(rows)
    metrics=["MSP","Energy","Entropy","KNN","Conformal","Mahalanobis"]
    summ={m:f"{df[m].mean():.3f} +/- {df[m].std(ddof=1):.3f}" for m in metrics}
    pd.concat([df,pd.DataFrame([{"seed":"MEAN+/-STD",**summ}])],ignore_index=True)\
      .to_csv(RESULTS/"openset_baselines.csv",index=False)
    print("\nAUROC vs unknown faults (mean +/- std, 5 seeds):")
    for m in metrics: print(f"  {m:12s}: {summ[m]}")
    print("\nSaved -> results/openset_baselines.csv")

if __name__=="__main__":
    main()
