"""STEP 04 — Class-imbalance robustness (rare/incipient faults).

Run:  python src/step04_imbalance.py   (needs step02 cache)

Compares a plain classifier vs a class-balanced one, reporting overall macro-F1
and, crucially, per-class recall on the hard incipient faults (RARE_FAULTS).

Outputs (results/):
  imbalance_metrics.csv      (overall)
  imbalance_rare_recall.csv  (recall on rare faults, plain vs balanced)
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score
from config import DATA, RESULTS, SEED, RARE_FAULTS

PROC = DATA / "tep_extended" / "processed"

def load():
    g = lambda n: np.load(PROC / n)
    flat = lambda A: A.reshape(A.shape[0], -1)
    return flat(g("X_train.npy")), g("y_train.npy"), flat(g("X_test.npy")), g("y_test.npy")

def main():
    Xtr, ytr, Xte, yte = load()
    pca = PCA(n_components=min(40, Xtr.shape[1]), random_state=SEED).fit(Xtr)
    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

    configs = {"plain": None, "balanced": "balanced"}
    overall, rare = [], []
    for name, cw in configs.items():
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED, class_weight=cw)
        clf.fit(Xtr, ytr); p = clf.predict(Xte)
        overall.append({"config": name,
                        "macro_f1": round(f1_score(yte, p, average="macro", zero_division=0), 4),
                        "weighted_f1": round(f1_score(yte, p, average="weighted", zero_division=0), 4)})
        row = {"config": name}
        for fault in RARE_FAULTS:
            mask = yte == fault
            row[f"recall_fault{fault}"] = round(recall_score(yte == fault, p == fault, zero_division=0), 4) if mask.any() else np.nan
        rare.append(row)
        print(f"  {name:9s} macroF1={overall[-1]['macro_f1']:.4f}  rare-recall={ {k:v for k,v in row.items() if k!='config'} }")

    pd.DataFrame(overall).to_csv(RESULTS / "imbalance_metrics.csv", index=False)
    pd.DataFrame(rare).to_csv(RESULTS / "imbalance_rare_recall.csv", index=False)
    print("Saved imbalance_metrics.csv, imbalance_rare_recall.csv")
    print("STEP 04 done.")

if __name__ == "__main__":
    main()
