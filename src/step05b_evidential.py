"""STEP 05b — Evidential deep learning: calibrated uncertainty + open-set.

Run:  python src/step05b_evidential.py   (needs torch + step02 cache)
Skipped automatically by run_all if torch is not installed.

Implements the Dirichlet/evidential head (Sensoy et al., 2018): the network emits
evidence -> Dirichlet(alpha); uncertainty mass u = K / sum(alpha) is an explicit
"I don't know" signal. Trained on KNOWN faults only (UNKNOWN_FAULTS held out).

Reports: closed-set accuracy, ECE, and open-set AUROC (u separates unknowns).
Outputs (results/):  evidential_metrics.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED, UNKNOWN_FAULTS
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch not installed. pip install torch")
from sklearn.metrics import accuracy_score, roc_auc_score

PROC = DATA / "tep_extended" / "processed"
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 20, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")

class EvNet(nn.Module):
    """CNN feature extractor + evidential (ReLU evidence) head."""
    def __init__(self, f, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(128, c)
    def forward(self, x):
        return self.head(self.body(x.transpose(1, 2)))  # logits -> evidence via relu outside

def edl_loss(logits, y_onehot, epoch):
    evidence = F.relu(logits)
    alpha = evidence + 1.0
    S = alpha.sum(1, keepdim=True)
    # Bayes-risk MSE term
    err = ((y_onehot - alpha / S) ** 2).sum(1, keepdim=True)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(1, keepdim=True)
    lam = min(1.0, epoch / 10.0)
    # KL to uniform Dirichlet on the mis-evidence (regularizer)
    a_tilde = y_onehot + (1 - y_onehot) * alpha
    K = alpha.shape[1]
    kl = (torch.lgamma(a_tilde.sum(1)) - torch.lgamma(torch.tensor(float(K)))
          - torch.lgamma(a_tilde).sum(1)
          + ((a_tilde - 1) * (torch.digamma(a_tilde) - torch.digamma(a_tilde.sum(1, keepdim=True)))).sum(1))
    return (err + var).mean() + lam * kl.mean()

def uncertainty(logits):
    alpha = F.relu(logits) + 1.0
    K = alpha.shape[1]
    return (K / alpha.sum(1)).cpu().numpy()          # vacuity / "I don't know" mass

def probs(logits):
    alpha = F.relu(logits) + 1.0
    return (alpha / alpha.sum(1, keepdim=True)).cpu().numpy()

def ece(P, preds, labels, n=10):
    conf = P.max(1); acc = (preds == labels).astype(float)
    b = np.linspace(0, 1, n + 1); e = 0.0
    for i in range(n):
        m = (conf > b[i]) & (conf <= b[i+1])
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def main():
    Xtr, ytr, Xte, yte = load()
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
    Xk, yk = Xtr[known], ytr[known]
    classes = np.unique(yk); cmap = {c: i for i, c in enumerate(classes)}
    enc = lambda y: np.array([cmap[v] for v in y])
    C = len(classes)
    dl = DataLoader(TensorDataset(torch.tensor(Xk), torch.tensor(enc(yk))), batch_size=BS, shuffle=True)

    net = EvNet(Xtr.shape[-1], C).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for ep in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            xb = xb.to(DEV); oh = F.one_hot(yb, C).float().to(DEV)
            opt.zero_grad(); loss = edl_loss(net(xb), oh, ep); loss.backward(); opt.step()

    net.eval()
    with torch.no_grad():
        logit_te = net(torch.tensor(Xte).to(DEV))
    P = probs(logit_te); u = uncertainty(logit_te)
    known_te = ~np.isin(yte, UNKNOWN_FAULTS)
    pred = classes[P.argmax(1)]
    acc = accuracy_score(yte[known_te], pred[known_te])
    cal = ece(P[known_te], pred[known_te], yte[known_te])
    is_unknown = np.isin(yte, UNKNOWN_FAULTS).astype(int)
    auroc = roc_auc_score(is_unknown, u) if is_unknown.sum() and (is_unknown == 0).sum() else np.nan

    df = pd.DataFrame([{"closed_set_acc": round(acc, 4), "ECE": round(cal, 4),
                        "open_set_AUROC": round(float(auroc), 4),
                        "unknown_faults": str(UNKNOWN_FAULTS), "device": DEV}])
    df.to_csv(RESULTS / "evidential_metrics.csv", index=False)
    print(df.to_string(index=False))
    print("STEP 05b done.")

if __name__ == "__main__":
    main()
