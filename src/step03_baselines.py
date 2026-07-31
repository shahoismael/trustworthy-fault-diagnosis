"""STEP 03 — Classical + ML baselines on the cached TEP splits.

Run:  python src/step03_baselines.py
Requires step02 outputs in data/tep_extended/processed/.

Baselines (all reproduced, not quoted):
  - PCA + Hotelling T^2 nearest-class (linear MSPC lineage)
  - Linear SVM
  - Random Forest
  - MLP (flattened window)
Deep baselines (1D-CNN / LSTM / Transformer) live in step03b (PyTorch).

Outputs (results/):
  baselines_metrics.csv   (accuracy, macro-F1, weighted-F1 per model)
"""
import json
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, f1_score
from config import DATA, RESULTS, SEED

PROC = DATA / "tep_extended" / "processed"

def load():
    def g(n): return np.load(PROC / n)
    Xtr, ytr = g("X_train.npy"), g("y_train.npy")
    Xva, yva = g("X_val.npy"),   g("y_val.npy")
    Xte, yte = g("X_test.npy"),  g("y_test.npy")
    # flatten windows for classical models: (N, W*F)
    flat = lambda A: A.reshape(A.shape[0], -1)
    return (flat(Xtr), ytr, flat(Xte), yte)

def evaluate(name, clf, Xtr, ytr, Xte, yte, rows):
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    rows.append({
        "model": name,
        "accuracy": round(accuracy_score(yte, p), 4),
        "macro_f1": round(f1_score(yte, p, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(yte, p, average="weighted", zero_division=0), 4),
    })
    print(f"  {name:16s} acc={rows[-1]['accuracy']:.4f}  macroF1={rows[-1]['macro_f1']:.4f}")

def main():
    Xtr, ytr, Xte, yte = load()
    print(f"train {Xtr.shape}  test {Xte.shape}  classes {np.unique(ytr).size}")
    rows = []

    # PCA -> NearestCentroid (linear MSPC-style baseline)
    k = min(30, Xtr.shape[1])
    pca = PCA(n_components=k, random_state=SEED).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    nc = NearestCentroid().fit(Ztr, ytr)
    p = nc.predict(Zte)
    rows.append({"model": "PCA+NearestCentroid",
                 "accuracy": round(accuracy_score(yte, p), 4),
                 "macro_f1": round(f1_score(yte, p, average="macro", zero_division=0), 4),
                 "weighted_f1": round(f1_score(yte, p, average="weighted", zero_division=0), 4)})
    print(f"  {'PCA+NC':16s} acc={rows[-1]['accuracy']:.4f}  macroF1={rows[-1]['macro_f1']:.4f}")

    # remaining baselines run on PCA-reduced features -> fast, same protocol
    evaluate("LinearSVM", LinearSVC(random_state=SEED, dual="auto", max_iter=5000), Ztr, ytr, Zte, yte, rows)
    evaluate("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED), Ztr, ytr, Zte, yte, rows)
    evaluate("MLP", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=SEED), Ztr, ytr, Zte, yte, rows)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "baselines_metrics.csv", index=False)
    print("\nSaved ->", RESULTS / "baselines_metrics.csv")
    print(df.to_string(index=False))
    print("\nSTEP 03 done. Next: step03b deep baselines (CNN/LSTM/Transformer).")

if __name__ == "__main__":
    main()
