"""STEP 17 — Ablation on the TEP CNN (fixes flaw #5, component isolation).

Run:  python src/step17_ablation.py   (needs torch + step02 cache)

Isolates each design choice on the SAME backbone and split:
  A. plain cross-entropy
  B. focal loss (no class weights)
  C. focal + sqrt class weights   (the model used in the paper)
Reports macro-F1 and rare-fault (3/9/15) recall for each -> shows what each
component contributes. Single seed (42) for a clean controlled comparison.

Outputs (results/):  ablation.csv
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED, RARE_FAULTS, UNKNOWN_FAULTS
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, recall_score

PROC = DATA / "tep_extended" / "processed"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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

def run(Xtr,ytr,Xte,yte,mode):
    torch.manual_seed(SEED); np.random.seed(SEED)
    known=~np.isin(ytr,UNKNOWN_FAULTS)
    kcl=np.unique(ytr[known]); km={c:i for i,c in enumerate(kcl)}; C=len(kcl)
    enc=lambda y:np.array([km[v] for v in y])
    cnt=np.bincount(enc(ytr[known]),minlength=C).astype(float)
    w=torch.tensor(np.sqrt(cnt.sum()/(C*np.clip(cnt,1,None))),dtype=torch.float32,device=DEV) if mode=="focal+weight" else None
    dl=DataLoader(TensorDataset(torch.tensor(Xtr[known]),torch.tensor(enc(ytr[known]),dtype=torch.long)),batch_size=BS,shuffle=True)
    net=CNN(Xtr.shape[-1],C).to(DEV); opt=torch.optim.Adam(net.parameters(),lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for xb,yb in dl:
            xb,yb=xb.to(DEV),yb.to(DEV); opt.zero_grad()
            out=net(xb); loss=F.cross_entropy(out,yb) if mode=="plain-CE" else focal(out,yb,w)
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pr=torch.cat([net(torch.tensor(Xte[i:i+512]).to(DEV)).cpu() for i in range(0,len(Xte),512)]).argmax(1).numpy()
    pred=kcl[pr]; kte=~np.isin(yte,UNKNOWN_FAULTS)
    r={"config":mode,"macro_f1":round(f1_score(yte[kte],pred[kte],average="macro",zero_division=0),4)}
    for fn in RARE_FAULTS: r[f"recall_f{fn}"]=round(recall_score(yte==fn,pred==fn,zero_division=0),4)
    return r

def main():
    Xtr,ytr,Xte,yte=load()
    rows=[run(Xtr,ytr,Xte,yte,m) for m in ["plain-CE","focal","focal+weight"]]
    for r in rows: print(r)
    pd.DataFrame(rows).to_csv(RESULTS/"ablation.csv",index=False)
    print("Saved ablation.csv"); print("STEP 17 done.")

if __name__ == "__main__":
    main()
