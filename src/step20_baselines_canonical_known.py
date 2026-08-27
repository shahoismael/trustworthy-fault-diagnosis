"""STEP 20 — Baselines on the CANONICAL test set, KNOWN classes only.

Purpose: make Table 2 apples-to-apples with the unified CNN, which is trained
and evaluated on the known classes only (unknown faults 16/17/18 held out).
This reruns the shallow baselines under the identical class set and the same
canonical train/test protocol (train on *_Training, test on *_Testing).

Run from anywhere in the repo:
    python submission-TIMC/step20_baselines_canonical_known.py

Output: results/baselines_canonical_known.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "tep_extended" / "processed"
RESULTS = ROOT / "results"
SEED = 42
UNKNOWN = [16, 17, 18]          # held out of BOTH train and test to match the unified model

def load():
    g = lambda n: np.load(PROC / n)
    Xtr, ytr, Xte, yte = g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")
    ktr, kte = ~np.isin(ytr, UNKNOWN), ~np.isin(yte, UNKNOWN)   # known classes only
    flat = lambda A: A.reshape(A.shape[0], -1)
    return flat(Xtr[ktr]), ytr[ktr], flat(Xte[kte]), yte[kte]

def row(name, yte, p):
    return {"model": name,
            "accuracy": round(accuracy_score(yte, p), 4),
            "macro_f1": round(f1_score(yte, p, average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(yte, p, average="weighted", zero_division=0), 4)}

def main():
    Xtr, ytr, Xte, yte = load()
    print(f"KNOWN-class canonical: train {Xtr.shape}  test {Xte.shape}  classes {np.unique(ytr).size}")
    k = min(30, Xtr.shape[1])
    pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    rows = []
    nc = NearestCentroid().fit(Ztr, ytr)
    rows.append(row("PCA+NearestCentroid", yte, nc.predict(Zte)))
    for name, clf in [
        ("LinearSVM", LinearSVC(random_state=SEED, dual="auto", max_iter=5000)),
        ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=SEED)),
    ]:
        clf.fit(Ztr, ytr)
        rows.append(row(name, yte, clf.predict(Zte)))
    for r in rows:
        print(f"  {r['model']:20s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baselines_canonical_known.csv", index=False)
    print("\nSaved ->", RESULTS / "baselines_canonical_known.csv")

if __name__ == "__main__":
    main()
