"""STEP 16 — Multi-seed CIs for the auxiliary experiments (fixes flaw #4).

Run:  python src/step16_aux_multiseed.py

Runs Steel Plates (classification), Debutanizer + SRU (soft-sensor regression),
and IndPenSim (sample-level detection) over 5 seeds each and reports mean +/- std,
so every reported auxiliary number carries a confidence interval.

Outputs (results/):  aux_multiseed.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score, mean_squared_error
from config import STEEL_DIR, SOFTSENSOR_DIR, INDPENSIM_DIR, RESULTS

SEEDS = [42, 1, 2, 3, 4]
FAULTS = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]

def summ(name, metric, vals):
    return {"experiment": name, "metric": metric,
            "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n_seeds": len(vals)}

def lag_embed(U, y, L=5):
    rows = []
    for t in range(L, len(U)):
        f = [U[t-k] for k in range(0, L+1)] + [[y[t-k]] for k in range(1, L+1)]
        rows.append(np.concatenate([np.ravel(z) for z in f]))
    return np.array(rows), y[L:]

def main():
    rows = []

    # --- Steel Plates (classification) ---
    df = pd.read_csv(STEEL_DIR / "steel_plates_faults_original_dataset.csv")
    df.columns = [c.strip() for c in df.columns]
    faults = [c for c in FAULTS if c in df.columns]
    y = df[faults].to_numpy().argmax(1)
    drop = set(faults) | {"id","Id","ID"}
    X = df[[c for c in df.columns if c not in drop]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    accs, f1s = [], []
    for s in SEEDS:
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.3, random_state=s, stratify=y)
        p = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=s).fit(Xtr,ytr).predict(Xte)
        accs.append(accuracy_score(yte,p)); f1s.append(f1_score(yte,p,average="macro",zero_division=0))
    rows += [summ("Steel","accuracy",accs), summ("Steel","macro_f1",f1s)]

    # --- Soft sensors (regression, lagged) ---
    def sensor(name, U, yv):
        Xl, yl = lag_embed(U, yv, 5); n = int(len(Xl)*0.7)
        r2s, rmses = [], []
        for s in SEEDS:
            m = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=s).fit(Xl[:n], yl[:n])
            p = m.predict(Xl[n:]); r2s.append(r2_score(yl[n:],p)); rmses.append(np.sqrt(mean_squared_error(yl[n:],p)))
        return [summ(name,"R2",r2s), summ(name,"RMSE",rmses)]
    deb = pd.read_csv(SOFTSENSOR_DIR/"Debutanizer_Data.txt", sep=r"\s+", engine="python")
    deb.columns=[c.strip() for c in deb.columns]; yc=[c for c in deb.columns if c.lower()=="y"][-1]
    rows += sensor("Debutanizer", deb.drop(columns=[yc]).to_numpy(float), deb[yc].to_numpy(float))
    sru = pd.read_csv(SOFTSENSOR_DIR/"SRU.csv"); sru.columns=[c.strip().lstrip("﻿") for c in sru.columns]
    us=[c for c in sru.columns if c.startswith("u")]
    rows += sensor("SRU_y1", sru[us].to_numpy(float), sru["y1"].to_numpy(float))

    # --- IndPenSim (sample-level detection) ---
    cols = pd.read_csv(INDPENSIM_DIR/"100_Batches_IndPenSim_V3.csv", nrows=0).columns.tolist()
    fref = [c for c in cols if "fault_ref" in c.lower() or ("fault" in c.lower() and "ref" in c.lower())]
    if fref:
        fref=fref[0]; feats=cols[1:31]
        d = pd.read_csv(INDPENSIM_DIR/"100_Batches_IndPenSim_V3.csv", usecols=list(dict.fromkeys(feats+[fref])))
        yi=(pd.to_numeric(d[fref],errors="coerce").fillna(0)>0).astype(int).to_numpy()
        Xi=d[feats].apply(pd.to_numeric,errors="coerce").fillna(0).to_numpy(float)
        Xi=(Xi-Xi.mean(0))/(Xi.std(0)+1e-8)
        aucs=[]
        for s in SEEDS:
            idx=np.random.RandomState(s).choice(len(Xi), min(60000,len(Xi)), replace=False)
            cv=StratifiedKFold(5, shuffle=True, random_state=s)
            pr=cross_val_predict(RandomForestClassifier(150,n_jobs=-1,random_state=s,class_weight="balanced"),
                                 Xi[idx], yi[idx], cv=cv, method="predict_proba")[:,1]
            aucs.append(roc_auc_score(yi[idx], pr))
        rows += [summ("IndPenSim(sample)","AUROC",aucs)]

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS/"aux_multiseed.csv", index=False)
    print(out.to_string(index=False)); print("STEP 16 done.")

if __name__ == "__main__":
    main()
