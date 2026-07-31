"""STEP 14 — IndPenSim BATCH-LEVEL fault detection (dataset #3, fixed).

Run:  python src/step14_indpensim.py

A batch is faulty if it contains any injected fault (Fault_ref > 0). We aggregate
each batch to per-variable mean+std and classify batches as normal vs faulty.
This is the correct framing (per-sample flags gave a degenerate 99% fault rate).

Outputs (results/):  indpensim_metrics.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from config import INDPENSIM_DIR, RESULTS, SEED

CSV = INDPENSIM_DIR / "100_Batches_IndPenSim_V3.csv"

def find(cols, *keys):
    for c in cols:
        cl = c.lower()
        if all(k in cl for k in keys):
            return c
    return None

def main():
    cols = pd.read_csv(CSV, nrows=0).columns.tolist()
    fault_ref = find(cols, "fault_ref") or find(cols, "fault", "ref")
    batch_ref = find(cols, "batch id") or find(cols, "batch", "id") or find(cols, "batch_ref")
    if not fault_ref or not batch_ref:
        print("Missing Fault_ref/Batch ID; STEP 14 skipped."); return
    print("Grouping by:", batch_ref, "| label:", fault_ref)
    feats = cols[1:31]                                   # continuous process/measurement vars
    use = list(dict.fromkeys(feats + [fault_ref, batch_ref]))
    df = pd.read_csv(CSV, usecols=use)
    df = df.dropna(subset=[batch_ref])
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")

    g = df.groupby(batch_ref)
    Xmean = g[feats].mean(); Xstd = g[feats].std().fillna(0.0)
    X = np.hstack([Xmean.to_numpy(), Xstd.to_numpy()])
    y = (g[fault_ref].max() > 0).astype(int).to_numpy()   # batch faulty if any fault injected
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    minc = int(min(np.bincount(y))) if len(np.unique(y)) > 1 else 1
    nsplits = max(2, min(5, minc))
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")
    cv = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=SEED)
    pred = cross_val_predict(clf, X, y, cv=cv)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    row = [{"dataset": "IndPenSim (sample-level detection)", "n_units": int(len(y)),
            "fault_rate": round(float(y.mean()), 4),
            "accuracy": round(accuracy_score(y, pred), 4),
            "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 4),
            "AUROC": round(float(roc_auc_score(y, proba)), 4) if y.min() != y.max() else np.nan}]
    pd.DataFrame(row).to_csv(RESULTS / "indpensim_metrics.csv", index=False)
    print(row[0]); print("STEP 14 done.")

if __name__ == "__main__":
    main()
