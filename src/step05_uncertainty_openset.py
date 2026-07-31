"""STEP 05 — Calibrated uncertainty + open-set recognition.

Run:  python src/step05_uncertainty_openset.py   (needs step02 cache)

- Trains a classifier on KNOWN faults only (UNKNOWN_FAULTS held out entirely).
- Uncertainty = predictive entropy of class probabilities.
- Calibration = Expected Calibration Error (ECE) on known-fault test data.
- Open-set = can uncertainty separate held-out UNKNOWN faults from known? (AUROC)

Outputs (results/):
  uncertainty_openset_metrics.csv   (ECE, closed-set acc, open-set AUROC)

Note: this uses probability-entropy uncertainty (runs anywhere). The evidential
deep-learning upgrade (Sensoy et al., 2018) is the paper's headline variant and
lives in step05b (PyTorch) — same protocol, stronger calibration.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from config import DATA, RESULTS, SEED, UNKNOWN_FAULTS

PROC = DATA / "tep_extended" / "processed"

def load():
    g = lambda n: np.load(PROC / n)
    flat = lambda A: A.reshape(A.shape[0], -1)
    return flat(g("X_train.npy")), g("y_train.npy"), flat(g("X_test.npy")), g("y_test.npy")

def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)

def ece(probs, preds, labels, n_bins=10):
    conf = probs.max(1); acc = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i+1])
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def main():
    Xtr, ytr, Xte, yte = load()
    pca = PCA(n_components=min(40, Xtr.shape[1]), random_state=SEED).fit(Xtr)
    Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)

    known_tr = ~np.isin(ytr, UNKNOWN_FAULTS)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    clf.fit(Xtr[known_tr], ytr[known_tr])

    known_te = ~np.isin(yte, UNKNOWN_FAULTS)
    Pk = clf.predict_proba(Xte[known_te]); predk = clf.classes_[Pk.argmax(1)]
    acc = accuracy_score(yte[known_te], predk)
    cal = ece(Pk, predk, yte[known_te])

    # open-set: uncertainty should be higher for UNKNOWN than for known
    Pall = clf.predict_proba(Xte); u = entropy(Pall)
    is_unknown = np.isin(yte, UNKNOWN_FAULTS).astype(int)
    auroc = roc_auc_score(is_unknown, u) if is_unknown.sum() and (is_unknown == 0).sum() else np.nan

    df = pd.DataFrame([{
        "closed_set_acc": round(acc, 4),
        "ECE": round(cal, 4),
        "open_set_AUROC": round(auroc, 4),
        "unknown_faults": str(UNKNOWN_FAULTS),
    }])
    df.to_csv(RESULTS / "uncertainty_openset_metrics.csv", index=False)
    print(df.to_string(index=False))
    print("STEP 05 done.")

if __name__ == "__main__":
    main()
