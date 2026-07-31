"""STEP 10 — Multi-seed unified CNN (fixes 'single-seed' flaw #5).

Run:  python src/step10_multiseed.py   (needs torch + step02 cache)

Repeats the unified CNN (classification + focal + Mahalanobis open-set) over
several seeds and reports mean +/- std for every headline metric — the
statistical rigor Q1 reviewers require.

Outputs (results/):  multiseed_summary.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, RARE_FAULTS, UNKNOWN_FAULTS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score, roc_auc_score

PROC = DATA / "tep_extended" / "processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 1, 2]          # >=3 for mean+/-std; add more for the camera-ready
EPOCHS, BS, LR = 12, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")

class CNN(nn.Module):
    def __init__(s, f, c):
        super().__init__()
        s.body = nn.Sequential(nn.Conv1d(f,64,3,padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
                               nn.Conv1d(64,128,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3))
        s.head = nn.Linear(128, c)
    def features(s,x): return s.body(x.transpose(1,2))
    def forward(s,x):  return s.head(s.features(x))

def focal(lg, t, w, g=2.0):
    ce = F.cross_entropy(lg, t, weight=w, reduction="none"); return ((1-torch.exp(-ce))**g*ce).mean()

def one_seed(seed, Xtr, ytr, Xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
    kcl = np.unique(ytr[known]); kmap = {c:i for i,c in enumerate(kcl)}; C=len(kcl)
    enc = lambda y: np.array([kmap[v] for v in y])
    cnt = np.bincount(enc(ytr[known]), minlength=C).astype(float)
    w = torch.tensor(np.sqrt(cnt.sum()/(C*np.clip(cnt,1,None))), dtype=torch.float32, device=DEV)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr[known]), torch.tensor(enc(ytr[known]),dtype=torch.long)), batch_size=BS, shuffle=True)
    net = CNN(Xtr.shape[-1], C).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad(); focal(net(xb),yb,w).backward(); opt.step()
    net.eval()
    bat = lambda fn,X: torch.cat([fn(torch.tensor(X[i:i+512]).to(DEV)) for i in range(0,len(X),512)])
    with torch.no_grad():
        lg = bat(lambda t: net(t).cpu(), Xte)
        fte = bat(lambda t: net.features(t).cpu(), Xte).numpy()
        ftr = bat(lambda t: net.features(t).cpu(), Xtr[known]).numpy()
    P = F.softmax(lg,1).numpy(); pred = kcl[P.argmax(1)]
    kte = ~np.isin(yte, UNKNOWN_FAULTS)
    out = {"closed_macro_f1": f1_score(yte[kte], pred[kte], average="macro", zero_division=0)}
    for fn in RARE_FAULTS: out[f"recall_f{fn}"] = recall_score(yte==fn, pred==fn, zero_division=0)
    mus = np.stack([ftr[enc(ytr[known])==i].mean(0) for i in range(C)])
    cov = np.cov((ftr-mus[enc(ytr[known])]).T)+1e-6*np.eye(ftr.shape[1]); prec=np.linalg.pinv(cov)
    d = fte[:,None,:]-mus[None]; maha=np.einsum("ncd,de,nce->nc",d,prec,d).min(1)
    isu = np.isin(yte, UNKNOWN_FAULTS).astype(int)
    out["openset_maha_AUROC"] = roc_auc_score(isu, maha)
    return out

def main():
    Xtr,ytr,Xte,yte = load()
    recs = []
    for s in SEEDS:
        r = one_seed(s, Xtr, ytr, Xte, yte); r["seed"]=s; recs.append(r)
        print("seed", s, {k:round(v,4) for k,v in r.items() if k!='seed'})
    df = pd.DataFrame(recs)
    agg = df.drop(columns="seed").agg(["mean","std"]).T
    agg.columns = ["mean","std"]; agg = agg.round(4)
    agg.to_csv(RESULTS / "multiseed_summary.csv")
    print("\nMean +/- std over", len(SEEDS), "seeds:")
    print(agg.to_string())
    print("STEP 10 done.")

if __name__ == "__main__":
    main()
