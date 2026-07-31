"""STEP 03b — Deep baselines (1D-CNN, LSTM) in PyTorch.

Run:  python src/step03b_deep_baselines.py   (needs torch + step02 cache)
Skipped automatically by run_all if torch is not installed.

Outputs (results/):  deep_baselines_metrics.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED
try:
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch not installed. pip install torch")
from sklearn.metrics import accuracy_score, f1_score

PROC = DATA / "tep_extended" / "processed"
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return (g("X_train.npy"), g("y_train.npy"), g("X_val.npy"), g("y_val.npy"),
            g("X_test.npy"), g("y_test.npy"))

class CNN1D(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, c))
    def forward(self, x):            # x: (B, W, F) -> (B, F, W)
        return self.net(x.transpose(1, 2))

class LSTMNet(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.lstm = nn.LSTM(f, 128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, c))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1])

def run(model, tr, va, te, classes):
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in te:
            preds.append(model(xb.to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(preds)

def main():
    Xtr, ytr, Xval, yval, Xte, yte = load()
    classes = np.unique(ytr); cmap = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([cmap[v] for v in y])
    to = lambda X, y: DataLoader(TensorDataset(torch.tensor(X), torch.tensor(enc(y))),
                                 batch_size=BS, shuffle=True)
    tr, va, te = to(Xtr, ytr), to(Xval, yval), DataLoader(
        TensorDataset(torch.tensor(Xte), torch.tensor(enc(yte))), batch_size=BS)
    f, c = Xtr.shape[-1], len(classes)
    rows = []
    for name, Model in [("CNN1D", CNN1D), ("BiLSTM", LSTMNet)]:
        p = run(Model(f, c), tr, va, te, classes)
        inv = classes[p]
        rows.append({"model": name,
                     "accuracy": round(accuracy_score(yte, inv), 4),
                     "macro_f1": round(f1_score(yte, inv, average="macro", zero_division=0), 4),
                     "weighted_f1": round(f1_score(yte, inv, average="weighted", zero_division=0), 4)})
        print(f"  {name:8s} acc={rows[-1]['accuracy']:.4f}  macroF1={rows[-1]['macro_f1']:.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "deep_baselines_metrics.csv", index=False)
    print("Saved deep_baselines_metrics.csv  (device:", DEV, ")")
    print("STEP 03b done.")

if __name__ == "__main__":
    main()
