"""STEP 04b — Deep imbalance + open-set on the CNN backbone (Path A).

Run:  python src/step04b_deep_imbalance_openset.py   (needs torch + step02 cache)

Fixes the two weak pillars by moving them onto the strong CNN backbone:
  (1) Imbalance: CNN trained with FOCAL loss -> per-class recall on rare faults 3/9/15.
  (2) Open-set: CNN trained on KNOWN faults only; unknowns (16/17/18) scored by
      ENERGY (Liu et al., 2020) and softmax ENTROPY -> AUROC. Compared to the
      plain cross-entropy CNN.

Outputs (results/):
  deep_imbalance_rare_recall.csv
  deep_openset_metrics.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED, RARE_FAULTS, UNKNOWN_FAULTS
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch not installed. pip install torch")
from sklearn.metrics import f1_score, recall_score, roc_auc_score

PROC = DATA / "tep_extended" / "processed"
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")

class CNN(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3))
        self.head = nn.Linear(128, c)
    def forward(self, x):
        return self.head(self.body(x.transpose(1, 2)))

def focal_loss(logits, target, gamma=2.0, weight=None):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

def train(model, X, y, C, epochs=EPOCHS, use_focal=True, cw=None):
    dl = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)), batch_size=BS, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR); model.to(DEV).train()
    w = torch.tensor(cw, dtype=torch.float32, device=DEV) if cw is not None else None
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            out = model(xb)
            loss = focal_loss(out, yb, weight=w) if use_focal else F.cross_entropy(out, yb, weight=w)
            loss.backward(); opt.step()
    return model

def logits_of(model, X):
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 512):
            out.append(model(torch.tensor(X[i:i+512]).to(DEV)).cpu())
    return torch.cat(out)

def main():
    Xtr, ytr, Xte, yte = load()

    # ---------- (1) imbalance: focal CNN on all 21 classes ----------
    classes = np.unique(ytr); cmap = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([cmap[v] for v in y])
    cnt = np.bincount(enc(ytr), minlength=len(classes)).astype(float)
    cw = (cnt.sum() / (len(cnt) * np.clip(cnt, 1, None)))     # inverse-frequency weights
    m = train(CNN(Xtr.shape[-1], len(classes)), Xtr, enc(ytr), len(classes), use_focal=True, cw=cw)
    pred = classes[logits_of(m, Xte).argmax(1).numpy()]
    rows = [{"model": "CNN+focal",
             "macro_f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4)}]
    for fnum in RARE_FAULTS:
        rows[0][f"recall_fault{fnum}"] = round(recall_score(yte == fnum, pred == fnum, zero_division=0), 4)
    pd.DataFrame(rows).to_csv(RESULTS / "deep_imbalance_rare_recall.csv", index=False)
    print("Imbalance (CNN+focal):", rows[0])

    # ---------- (2) open-set: CNN on KNOWN faults; score unknowns ----------
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
    kcl = np.unique(ytr[known]); kmap = {c: i for i, c in enumerate(kcl)}
    kenc = lambda y: np.array([kmap[v] for v in y])
    mk = train(CNN(Xtr.shape[-1], len(kcl)), Xtr[known], kenc(ytr[known]), len(kcl), use_focal=False)
    lg = logits_of(mk, Xte)
    energy = (-torch.logsumexp(lg, dim=1)).numpy()            # higher -> OOD
    P = F.softmax(lg, dim=1).numpy(); ent = -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1)
    is_unknown = np.isin(yte, UNKNOWN_FAULTS).astype(int)
    ok = is_unknown.sum() and (is_unknown == 0).sum()
    out = pd.DataFrame([{
        "openset_energy_AUROC": round(roc_auc_score(is_unknown, energy), 4) if ok else np.nan,
        "openset_entropy_AUROC": round(roc_auc_score(is_unknown, ent), 4) if ok else np.nan,
        "unknown_faults": str(UNKNOWN_FAULTS), "device": DEV}])
    out.to_csv(RESULTS / "deep_openset_metrics.csv", index=False)
    print(out.to_string(index=False))
    print("STEP 04b done.")

if __name__ == "__main__":
    main()
