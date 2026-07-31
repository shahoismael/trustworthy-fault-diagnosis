"""STEP 12 — Soft-sensor regression on Debutanizer + SRU (uses dataset #4).

Run:  python src/step12_softsensor.py

Real refinery soft-sensor benchmark (Fortuna et al.). Predict hard-to-measure
quality variables from process inputs. Time-ordered split (no shuffling — these
are time series). Reports RMSE and R^2.

Outputs (results/):  softsensor_metrics.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from config import SOFTSENSOR_DIR, RESULTS, SEED

def lag_embed(U, y, L=5):
    """Dynamic features: current + L past steps of inputs, plus L past outputs.
    These processes have measurement delay, so instantaneous u->y fails."""
    n = len(U)
    rows = []
    for t in range(L, n):
        feat = [U[t - k] for k in range(0, L + 1)]        # u(t)..u(t-L)
        feat += [[y[t - k]] for k in range(1, L + 1)]     # past outputs only (no leakage)
        rows.append(np.concatenate([np.ravel(f) for f in feat]))
    return np.array(rows), y[L:]

def timesplit(X, y, frac=0.7):
    n = int(len(X) * frac); return X[:n], y[:n], X[n:], y[n:]

def eval_one(name, U, y, rows, L=5):
    X, y = lag_embed(U, y, L)
    Xtr, ytr, Xte, yte = timesplit(X, y)
    m = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    p = m.predict(Xte)
    rows.append({"dataset": name,
                 "RMSE": round(float(np.sqrt(mean_squared_error(yte, p))), 4),
                 "R2": round(float(r2_score(yte, p)), 4),
                 "n": len(X)})
    print(f"  {name:12s} RMSE={rows[-1]['RMSE']:.4f}  R2={rows[-1]['R2']:.4f}")

def main():
    rows = []
    # Debutanizer: whitespace-separated u1..u7 + y
    deb = pd.read_csv(SOFTSENSOR_DIR / "Debutanizer_Data.txt", sep=r"\s+", engine="python")
    deb.columns = [c.strip() for c in deb.columns]
    ycol = [c for c in deb.columns if c.lower() == "y"][-1]
    Xd = deb.drop(columns=[ycol]).to_numpy(float); yd = deb[ycol].to_numpy(float)
    eval_one("Debutanizer", Xd, yd, rows)

    # SRU: time,u1..u5,y1,y2  -> predict y1 (H2S)
    sru = pd.read_csv(SOFTSENSOR_DIR / "SRU.csv")
    sru.columns = [c.strip().lstrip("﻿") for c in sru.columns]
    us = [c for c in sru.columns if c.startswith("u")]
    eval_one("SRU (y1)", sru[us].to_numpy(float), sru["y1"].to_numpy(float), rows)

    pd.DataFrame(rows).to_csv(RESULTS / "softsensor_metrics.csv", index=False)
    print("Saved softsensor_metrics.csv")
    print("STEP 12 done.")

if __name__ == "__main__":
    main()
