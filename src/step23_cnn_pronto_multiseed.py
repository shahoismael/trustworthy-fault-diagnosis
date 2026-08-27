"""STEP 23 — Real PRONTO detector over multiple seeds (adds dispersion to Table 6).

Fixes the reviewer point that the real-plant headline (AUROC ~0.99) was a single
run while the rest of the paper reports 5-seed intervals. Same architecture, same
time-ordered split as step15; only the network initialization/training seed varies.

Run from anywhere in the repo:
    python submission-TIMC/step23_cnn_pronto_multiseed.py

Output: results/cnn_pronto_multiseed.csv
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from step06_sim2real_pronto import load_merged, align   # reuse validated PRONTO loader

RESULTS = ROOT / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 20, 64, 1e-3
SEEDS = [42, 7, 123, 2024, 2025]

class CNN(nn.Module):
    def __init__(s, f, c=2):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
                              nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                              nn.Dropout(0.3), nn.Linear(128, c))
    def forward(s, x): return s.net(x.transpose(1, 2))

def order_split(X, y, frac=0.7):
    tr_i, te_i = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]; k = int(len(idx) * frac)
        tr_i += list(idx[:k]); te_i += list(idx[k:])
    return np.array(tr_i), np.array(te_i)

def run(seed, Xtr, ytr, Xte, yte, nf):
    torch.manual_seed(seed); np.random.seed(seed)
    net = CNN(nf).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
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
    return {"seed": seed,
            "accuracy": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
            "AUROC": float(roc_auc_score(yte, P[:, 1]))}

def main():
    normal, fault = load_merged()
    if not normal or not fault:
        print("PRONTO not available; STEP 23 skipped."); return
    Xn, common = align(normal); Xf, _ = align(fault)
    X = np.concatenate([Xn, Xf]).astype(np.float32)
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xf))].astype(int)
    tr, te = order_split(X, y)
    m = X[tr].reshape(-1, X.shape[-1]).mean(0); s = X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-8
    Xtr = ((X[tr] - m) / s).astype(np.float32); Xte = ((X[te] - m) / s).astype(np.float32)
    ytr, yte = y[tr], y[te]

    rows = [run(sd, Xtr, ytr, Xte, yte, X.shape[-1]) for sd in SEEDS]
    for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    df = pd.DataFrame(rows)
    summ = {"seed": "MEAN+/-STD"}
    for m_ in ["accuracy", "macro_f1", "AUROC"]:
        summ[m_] = f"{df[m_].mean():.3f} +/- {df[m_].std(ddof=1):.3f}"
    pd.concat([df, pd.DataFrame([summ])], ignore_index=True).to_csv(RESULTS / "cnn_pronto_multiseed.csv", index=False)
    print("\n5-seed PRONTO:", {k: summ[k] for k in ["accuracy", "macro_f1", "AUROC"]})
    print("Saved ->", RESULTS / "cnn_pronto_multiseed.csv")

if __name__ == "__main__":
    main()
