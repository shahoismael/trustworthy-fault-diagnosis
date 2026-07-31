"""STEP 06 — Method-level simulation-to-real transfer (TEP -> PRONTO).

Run:  python src/step06_sim2real_pronto.py

Locked design (DECISIONS.md, option a): transfer the DETECTOR METHOD, not weights.
  1. Detector methodology = PCA reconstruction-error monitor (established on TEP normal).
  2. Re-fit the SAME method on PRONTO NORMAL data (real plant).
  3. Score PRONTO fault windows; report normal-vs-fault separability (AUROC).

Uses the long merged PRONTO files (Testday*_merged.csv), which carry a 'Fault'
label column and 17 continuous process variables.

Outputs (results/):  sim2real_metrics.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from config import (DATA, RESULTS, SEED, WINDOW, STRIDE,
                    PRONTO_ALIGNED, PRONTO_MERGED_GLOB, PRONTO_LABEL, PRONTO_PROC_VARS)

# Explicit PRONTO label semantics (validated against merged files):
#   normal = 'Normal'; seeded faults = air blockage / air leakage / diverted flow.
#   'Slugging' is an operating regime (not a seeded fault) -> excluded.
NORMAL_TOKENS = {"normal"}
FAULT_TOKENS = {"air blockage", "air leakage", "diverted flow"}

def windows(a):
    return (np.stack([a[i:i+WINDOW] for i in range(0, len(a)-WINDOW+1, STRIDE)])
            if len(a) >= WINDOW else np.empty((0, WINDOW, a.shape[1] if a.ndim > 1 else len(PRONTO_PROC_VARS))))

def load_merged():
    base = Path(PRONTO_ALIGNED)
    normal, fault = [], []
    for f in sorted(base.glob(PRONTO_MERGED_GLOB)):
        df = pd.read_csv(f)
        df.columns = [str(c).strip() for c in df.columns]
        vars_here = [c for c in PRONTO_PROC_VARS if c in df.columns]
        if PRONTO_LABEL not in df.columns or len(vars_here) < 8:
            continue
        X = df[vars_here].apply(pd.to_numeric, errors="coerce")
        lab = df[PRONTO_LABEL].astype(str).str.strip().str.lower()
        Xn = X[lab.isin(NORMAL_TOKENS)].dropna().to_numpy(np.float32)
        Xf = X[lab.isin(FAULT_TOKENS)].dropna().to_numpy(np.float32)   # Slugging excluded
        # keep a common variable ordering across files
        if Xn.shape[1] == len(vars_here):
            normal.append((vars_here, Xn)); fault.append((vars_here, Xf))
    return normal, fault

def align(records):
    """Intersect variable sets across files so widths match; return stacked windows."""
    if not records:
        return np.empty((0, WINDOW, 0))
    common = set(records[0][0])
    for v, _ in records:
        common &= set(v)
    common = [c for c in PRONTO_PROC_VARS if c in common]
    out = []
    for v, X in records:
        idx = [v.index(c) for c in common]
        out.append(windows(X[:, idx]))
    out = [w for w in out if len(w)]
    return np.concatenate(out) if out else np.empty((0, WINDOW, len(common))), common

def main():
    normal, fault = load_merged()
    if not normal:
        print("No PRONTO merged files parsed; check config paths."); print("STEP 06 skipped."); return
    Xn, common = align(normal)
    Xf, _ = align(fault)
    if len(Xn) < 20:
        print(f"Too few PRONTO normal windows ({len(Xn)})."); print("STEP 06 skipped."); return

    flat = lambda A: A.reshape(A.shape[0], -1)
    mu = flat(Xn).mean(0); sd = flat(Xn).std(0) + 1e-8
    Xn_s = (flat(Xn) - mu) / sd
    split = int(0.7 * len(Xn_s))
    k = max(2, min(15, Xn_s.shape[1], split - 1))
    pca = PCA(n_components=k, random_state=SEED).fit(Xn_s[:split])
    recon = lambda Z: ((Z - pca.inverse_transform(pca.transform(Z))) ** 2).mean(1)

    e_norm = recon(Xn_s[split:])
    rows = [{"pronto_vars": len(common), "normal_windows": int(len(Xn)), "fault_windows": int(len(Xf))}]
    if len(Xf):
        Xf_s = (flat(Xf) - mu) / sd
        e_fault = recon(Xf_s)
        y = np.r_[np.zeros(len(e_norm)), np.ones(len(e_fault))]
        # --- naive method transfer: raw reconstruction error ---
        s_naive = np.r_[e_norm, e_fault]
        rows[0]["naive_AUROC"] = round(roc_auc_score(y, s_naive), 4)

        # --- adapted (option-a, few-shot target calibration) ---
        # score = Mahalanobis distance in whitened PCA space (variance-aware),
        # then a logistic calibrator fit on 30% labelled target, evaluated on 70%.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        Zn = pca.transform(Xn_s[split:]); Zf = pca.transform(Xf_s)
        var = pca.explained_variance_ + 1e-8
        maha = lambda Z: (Z ** 2 / var).sum(1)
        s_adapt = np.r_[maha(Zn), maha(Zf)].reshape(-1, 1)
        # few-shot target calibration, scored with 5-fold cross_val_predict
        # (out-of-fold -> no train/eval overlap, always well-defined)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = cross_val_predict(LogisticRegression(max_iter=1000), s_adapt, y,
                                cv=cv, method="predict_proba")[:, 1]
        rows[0]["adapted_AUROC"] = round(roc_auc_score(y, oof), 4)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sim2real_metrics.csv", index=False)
    print(df.to_string(index=False))
    print("STEP 06 done.")

if __name__ == "__main__":
    main()
