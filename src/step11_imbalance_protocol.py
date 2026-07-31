"""STEP 11 — GENUINE class-imbalance protocol (fixes flaw #1).

Run:  python src/step11_imbalance_protocol.py   (needs torch + step02 cache)

The base TEP training set is balanced, so 'imbalance robustness' was never really
tested. Here we IMPOSE imbalance: rare faults 3/9/15 are subsampled to IMB_FRAC of
their windows in TRAIN only (test stays untouched). We then compare a plain
cross-entropy CNN against a focal + class-weighted CNN, reporting rare-fault recall
under the imbalanced regime.

Outputs (results/):  imbalance_protocol.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED, RARE_FAULTS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score

PROC = DATA / "tep_extended" / "processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMB_FRAC = 0.05          # keep only 5% of rare-fault training windows
EPOCHS, BS, LR = 12, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")

class CNN(nn.Module):
    def __init__(s,f,c):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(f,64,3,padding=1),nn.ReLU(),nn.BatchNorm1d(64),nn.MaxPool1d(2),
                              nn.Conv1d(64,128,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),
                              nn.Dropout(0.3),nn.Linear(128,c))
    def forward(s,x): return s.net(x.transpose(1,2))

def focal(lg,t,w,g=2.0):
    ce=F.cross_entropy(lg,t,weight=w,reduction="none"); return ((1-torch.exp(-ce))**g*ce).mean()

def make_imbalanced(Xtr, ytr):
    rng = np.random.RandomState(SEED); keep = np.ones(len(ytr), bool)
    for fn in RARE_FAULTS:
        idx = np.where(ytr == fn)[0]
        drop = rng.choice(idx, int(len(idx) * (1 - IMB_FRAC)), replace=False)
        keep[drop] = False
    return Xtr[keep], ytr[keep]

def train_eval(Xtr, ytr, Xte, yte, use_focal):
    torch.manual_seed(SEED); np.random.seed(SEED)
    cl = np.unique(ytr); cm = {c:i for i,c in enumerate(cl)}; C=len(cl)
    enc = lambda y: np.array([cm[v] for v in y])
    cnt = np.bincount(enc(ytr), minlength=C).astype(float)
    w = torch.tensor(cnt.sum()/(C*np.clip(cnt,1,None)), dtype=torch.float32, device=DEV) if use_focal else None
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(enc(ytr),dtype=torch.long)), batch_size=BS, shuffle=True)
    net = CNN(Xtr.shape[-1], C).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad()
            out=net(xb); loss = focal(out,yb,w) if use_focal else F.cross_entropy(out,yb)
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr = torch.cat([net(torch.tensor(Xte[i:i+512]).to(DEV)).cpu() for i in range(0,len(Xte),512)]).argmax(1).numpy()
    pred = cl[pr]
    r = {"macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4)}
    for fn in RARE_FAULTS: r[f"recall_f{fn}"] = round(recall_score(yte==fn, pred==fn, zero_division=0), 4)
    return r

def main():
    Xtr, ytr, Xte, yte = load()
    Xi, yi = make_imbalanced(Xtr, ytr)
    print(f"Imposed imbalance: rare faults kept at {IMB_FRAC:.0%}. train {len(ytr)} -> {len(yi)}")
    rows = []
    for name, focal_flag in [("plain-CE", False), ("focal+weighted", True)]:
        r = train_eval(Xi, yi, Xte, yte, focal_flag); r["config"] = name; rows.append(r)
        print(f"  {name:16s} {r}")
    pd.DataFrame(rows).to_csv(RESULTS / "imbalance_protocol.csv", index=False)
    print("Saved imbalance_protocol.csv"); print("STEP 11 done.")

if __name__ == "__main__":
    main()
