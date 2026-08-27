"""STEP 21 — PCA T^2 / SPE as an open-set rejector (cheap MSPC comparator).

Purpose: answer the obvious referee question — does classical multivariate SPC
(Hotelling's T^2 and the squared prediction error, SPE/Q) reject unknown faults
as well as the network's feature-space scores? Fits PCA on NORMAL training
windows, then scores test windows; higher statistic = more anomalous. Reports
AUROC separating held-out unknown faults (16/17/18) from known classes.

Run from anywhere in the repo:
    python submission-TIMC/step21_pca_openset.py

Output: results/pca_openset.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "tep_extended" / "processed"
RESULTS = ROOT / "results"
SEED = 42
UNKNOWN = [16, 17, 18]
VAR_KEPT = 0.95                 # PCA components retaining 95% variance (standard MSPC choice)

def main():
    g = lambda n: np.load(PROC / n)
    Xtr, ytr, Xte, yte = g("X_train.npy"), g("y_train.npy"), g("X_test.npy"), g("y_test.npy")
    flat = lambda A: A.reshape(A.shape[0], -1)
    Xtr, Xte = flat(Xtr), flat(Xte)

    # fit PCA on NORMAL training windows only (in-control model)
    Xn = Xtr[ytr == 0]
    pca = PCA(n_components=VAR_KEPT, svd_solver="full", random_state=SEED).fit(Xn)
    lam = pca.explained_variance_                      # eigenvalues of retained PCs
    Zte = pca.transform(Xte)                           # scores in PC space

    # Hotelling's T^2 = sum_j score_j^2 / lambda_j
    t2 = (Zte ** 2 / lam).sum(1)
    # SPE / Q = squared reconstruction error in residual space
    Xhat = pca.inverse_transform(Zte)
    spe = ((Xte - Xhat) ** 2).sum(1)

    is_unknown = np.isin(yte, UNKNOWN).astype(int)
    out = pd.DataFrame([{
        "n_components": int(pca.n_components_),
        "T2_AUROC": round(roc_auc_score(is_unknown, t2), 4),
        "SPE_AUROC": round(roc_auc_score(is_unknown, spe), 4),
        "unknown_faults": str(UNKNOWN),
    }])
    out.to_csv(RESULTS / "pca_openset.csv", index=False)
    print(out.to_string(index=False))
    print("\nSaved ->", RESULTS / "pca_openset.csv")

if __name__ == "__main__":
    main()
