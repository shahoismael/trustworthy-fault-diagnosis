"""STEP 15 — Same CNN backbone on the REAL PRONTO plant (fixes 'unified' flaw #2).

Run:  python src/step15_cnn_pronto.py   (needs torch)

Applies the identical 1D-CNN used for TEP to real multiphase-flow plant data
(PRONTO), as a supervised normal-vs-fault detector. This makes the 'unified deep
framework' claim true across a SIMULATED process (TEP) and a REAL process (PRONTO)
-- distinct from the sim->real transfer experiment (which is reported separately as
a negative result). Time-ordered split (no shuffling) avoids adjacency leakage.

Outputs (results/):  cnn_pronto_metrics.csv
"""
import numpy as np, pandas as pd
from config import RESULTS, SEED, WINDOW
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from step06_sim2real_pronto import load_merged, align   # reuse validated PRONTO loader

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 20, 64, 1e-3
torch.manual_seed(SEED); np.random.seed(SEED)

class CNN(nn.Module):
    def __init__(s, f, c=2):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
                              nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                              nn.Dropout(0.3), nn.Linear(128, c))
    def forward(s, x): return s.net(x.transpose(1, 2))

def order_split(X, y, frac=0.7):
    # split each class by time order, then concatenate -> no shuffle leakage
    tr_i, te_i = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; k = int(len(idx) * frac)
        tr_i += list(idx[:k]); te_i += list(idx[k:])
    return np.array(tr_i), np.array(te_i)

def main():
    normal, fault = load_merged()
    if not normal or not fault:
        print("PRONTO not available; STEP 15 skipped."); return
    Xn, common = align(normal); Xf, _ = align(fault)
    X = np.concatenate([Xn, Xf]).astype(np.float32)
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xf))].astype(int)
    tr, te = order_split(X, y)
    m = X[tr].reshape(-1, X.shape[-1]).mean(0); s = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xtr = ((X[tr] - m) / s).astype(np.float32); Xte = ((X[te] - m) / s).astype(np.float32)
    ytr, yte = y[tr], y[te]

    net = CNN(X.shape[-1]).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
    cntw = np.bincount(ytr); w = torch.tensor(cntw.sum() / (2 * np.clip(cntw, 1, None)), dtype=torch.float32, device=DEV)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)), batch_size=BS, shuffle=True)
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV); opt.zero_grad()
            F.cross_entropy(net(xb), yb, weight=w).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        lg = torch.cat([net(torch.tensor(Xte[i:i+256]).to(DEV)).cpu() for i in range(0, len(Xte), 256)])
    P = F.softmax(lg, 1).numpy(); pred = P.argmax(1)
    row = [{"dataset": "PRONTO (real, same CNN)", "vars": len(common),
            "n_train": len(tr), "n_test": len(te),
            "accuracy": round(accuracy_score(yte, pred), 4),
            "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
            "AUROC": round(float(roc_auc_score(yte, P[:, 1])), 4) if yte.min() != yte.max() else np.nan}]
    pd.DataFrame(row).to_csv(RESULTS / "cnn_pronto_metrics.csv", index=False)
    print(row[0]); print("STEP 15 done.")

if __name__ == "__main__":
    main()
