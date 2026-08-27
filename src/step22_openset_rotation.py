"""STEP 22 — Open-set robustness: ROTATE the held-out unknown-fault set.

Purpose: the single triple (16/17/18) is the reviewers' top open-set concern.
This retrains the SAME unified CNN with several DIFFERENT held-out triples and
recomputes every rejection score, so the paper can report the open-set AUROC as
a mean over held-out sets rather than one lucky split.

This is the long run (many CNN trainings). Leave it running.
    python submission-TIMC/step22_openset_rotation.py

Edit FOLDS / SEEDS below to trade coverage for time.
Output: results/openset_rotation.csv  (per fold x seed rows + summary)
"""
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "tep_extended" / "processed"
RESULTS = ROOT / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3

# incipient faults (3,9,15) are kept as KNOWN in every fold; unknown triples are
# drawn from the remaining faults so each fold is a clean held-out set.
FOLDS = [[16, 17, 18], [1, 2, 4], [5, 6, 7], [8, 10, 11], [12, 13, 14]]
SEEDS = [42, 7, 123]            # 5 folds x 3 seeds = 15 trainings (~1.5-2 h on CPU)

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

def run(seed, unknown, data):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xval, yval, Xte, yte = data
    known = ~np.isin(ytr, unknown)
    kcl = np.unique(ytr[known]); kmap = {c: i for i, c in enumerate(kcl)}
    enc = lambda y: np.array([kmap[v] for v in y]); C = len(kcl)
    cnt = np.bincount(enc(ytr[known]), minlength=C).astype(float)
    w = torch.tensor(np.sqrt(cnt.sum() / (C * np.clip(cnt, 1, None))), dtype=torch.float32, device=DEV)
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
        logit_te = batch(lambda t: net(t).cpu(), Xte)
        feat_te = batch(lambda t: net.features(t).cpu(), Xte).numpy()
        feat_tr = batch(lambda t: net.features(t).cpu(), Xtr[known]).numpy()
        logit_val = batch(lambda t: net(t).cpu(), Xval)
    P = F.softmax(logit_te, 1).numpy()
    is_unknown = np.isin(yte, unknown).astype(int)

    msp = -P.max(1)
    energy = (-torch.logsumexp(logit_te, 1)).numpy()
    ent = -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1)
    ytr_k = enc(ytr[known])
    mus = np.stack([feat_tr[ytr_k == i].mean(0) for i in range(C)])
    cov = np.cov((feat_tr - mus[ytr_k]).T) + 1e-6 * np.eye(feat_tr.shape[1])
    prec = np.linalg.pinv(cov)
    d = feat_te[:, None, :] - mus[None]
    maha = np.einsum("ncd,de,nce->nc", d, prec, d).min(1)
    knn = NearestNeighbors(n_neighbors=5).fit(feat_tr).kneighbors(feat_te, return_distance=True)[0].mean(1)
    Pval = F.softmax(logit_val, 1).numpy()
    calib = 1.0 - Pval[~np.isin(yval, unknown)].max(1)
    test_nc = 1.0 - P.max(1)
    conf = 1.0 - (1.0 + (calib[None, :] >= test_nc[:, None]).sum(1)) / (len(calib) + 1.0)

    A = lambda s: round(roc_auc_score(is_unknown, s), 4)
    return {"fold": str(unknown), "seed": seed, "MSP": A(msp), "Energy": A(energy),
            "Entropy": A(ent), "KNN": A(knn), "Conformal": A(conf), "Mahalanobis": A(maha)}

def main():
    data = load()
    rows = []
    for fold in FOLDS:
        for s in SEEDS:
            r = run(s, fold, data); rows.append(r)
            print(r)
    df = pd.DataFrame(rows)
    metrics = ["MSP", "Energy", "Entropy", "KNN", "Conformal", "Mahalanobis"]
    summ = {"fold": "MEAN+/-STD (all folds/seeds)", "seed": ""}
    for m in metrics:
        summ[m] = f"{df[m].mean():.3f} +/- {df[m].std(ddof=1):.3f}"
    pd.concat([df, pd.DataFrame([summ])], ignore_index=True).to_csv(RESULTS / "openset_rotation.csv", index=False)
    print("\nOpen-set AUROC across", len(FOLDS), "held-out sets x", len(SEEDS), "seeds:")
    for m in metrics:
        print(f"  {m:12s}: {summ[m]}")
    print("\nSaved ->", RESULTS / "openset_rotation.csv")

if __name__ == "__main__":
    main()
