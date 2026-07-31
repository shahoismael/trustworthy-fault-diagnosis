"""STEP 07 — Interpretability: which process variables drive each fault.

Run:  python src/step07_interpretability.py   (needs step02 cache)

Trains a Random Forest on the 52 TEP variables (window-averaged so importances
map back to named variables), then reports global and per-fault top variables via
impurity importance + one-vs-rest signal. Cross-check against process topology.

Outputs (results/):
  interpretability_global_importance.csv
  interpretability_per_fault_top.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import DATA, RESULTS, SEED, TEP_FEATURES

PROC = DATA / "tep_extended" / "processed"

def load():
    g = lambda n: np.load(PROC / n)
    # window-average -> (N, 52) so importances are per named variable
    avg = lambda A: A.mean(1)
    return avg(g("X_train.npy")), g("y_train.npy"), avg(g("X_test.npy")), g("y_test.npy")

def main():
    Xtr, ytr, Xte, yte = load()
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)
    clf.fit(Xtr, ytr)

    imp = pd.DataFrame({"variable": TEP_FEATURES, "importance": clf.feature_importances_}) \
            .sort_values("importance", ascending=False)
    imp.to_csv(RESULTS / "interpretability_global_importance.csv", index=False)

    # per-fault: mean standardized deviation from normal (class 0) on top variables
    rows = []
    normal_mean = Xtr[ytr == 0].mean(0)
    normal_std = Xtr[ytr == 0].std(0) + 1e-8
    for fault in sorted(np.unique(ytr)):
        if fault == 0:
            continue
        dev = np.abs((Xtr[ytr == fault].mean(0) - normal_mean) / normal_std)
        top = np.argsort(dev)[::-1][:5]
        rows.append({"fault": int(fault),
                     "top_variables": ", ".join(f"{TEP_FEATURES[i]}({dev[i]:.1f}sd)" for i in top)})
    pd.DataFrame(rows).to_csv(RESULTS / "interpretability_per_fault_top.csv", index=False)
    print("Top-8 global variables:\n", imp.head(8).to_string(index=False))
    print("Saved interpretability_global_importance.csv, interpretability_per_fault_top.csv")
    print("STEP 07 done.")

if __name__ == "__main__":
    main()
