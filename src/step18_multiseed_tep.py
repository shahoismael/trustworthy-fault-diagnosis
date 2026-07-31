"""STEP 18 — Multi-seed classification on canonical TEP (adds CI to macro-F1).

Run:  python src/step18_multiseed_tep.py

Trains the SAME unified CNN as step09 across N seeds and reports
macro-F1 and rare-fault recall as mean ± std + 95% CI.

Output (results/):
  tep_multiseed_classification.csv   per-seed rows + summary
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, UNKNOWN_FAULTS, RARE_FAULTS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score

PROC = DATA / "tep_extended" / "processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3
SEEDS = [42, 7, 123, 2024, 2025]

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
    def forward(self, x): return self.head(self.body(x.transpose(1, 2)))

def focal(logits, target, gamma=2.0, w=None):
    ce = F.cross_entropy(logits, target, weight=w, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()

def run_seed(seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
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
    with torch.no_grad():
        out = []
        for i in range(0, len(Xte), 512):
            out.append(net(torch.tensor(Xte[i:i+512]).to(DEV)).cpu())
        P = F.softmax(torch.cat(out), 1).numpy()
    pred = kcl[P.argmax(1)]
    kt = ~np.isin(yte, UNKNOWN_FAULTS)
    row = {"seed": seed,
           "macro_f1": f1_score(yte[kt], pred[kt], average="macro", zero_division=0)}
    for fn in RARE_FAULTS:
        row[f"recall_f{fn}"] = recall_score(yte == fn, pred == fn, zero_division=0)
    return row

def main():
    Xtr, ytr, _, _, Xte, yte = load()
    rows = []
    for s in SEEDS:
        r = run_seed(s, Xtr, ytr, Xte, yte); rows.append(r)
        print(f"seed {s}: macro-F1={r['macro_f1']:.4f}")
    df = pd.DataFrame(rows)
    metrics = ["macro_f1"] + [f"recall_f{fn}" for fn in RARE_FAULTS]
    summ = {}
    for m in metrics:
        mu, sd = df[m].mean(), df[m].std(ddof=1)
        ci = 1.96 * sd / np.sqrt(len(df))
        summ[m] = f"{mu:.3f} ± {sd:.3f} (95% CI ±{ci:.3f})"
    out = pd.concat([df, pd.DataFrame([{"seed": "SUMMARY", **summ}])], ignore_index=True)
    out.to_csv(RESULTS / "tep_multiseed_classification.csv", index=False)
    print("\nSUMMARY:")
    for m in metrics: print(f"  {m}: {summ[m]}")
    print("\nSaved -> results/tep_multiseed_classification.csv")

if __name__ == "__main__":
    main()
