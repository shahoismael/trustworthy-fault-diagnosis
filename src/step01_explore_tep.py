"""STEP 01 — Explore the TEP dataset and produce first artefacts.

Run:  python src/step01_explore_tep.py
Outputs (to results/):
  - tep_class_distribution.csv
  - tep_class_distribution.png
Prints shapes, feature count, class balance, missing values.

Uses a small max_runs so it finishes in ~1-2 min. Raise it for the real runs.
"""
import numpy as np
import pandas as pd
from config import RESULTS, WINDOW, STRIDE
from tep_loader import load_tep

MAX_RUNS = 10  # start small; increase to 100+ for full experiments

def main():
    print(f"Loading TEP training windows (max_runs={MAX_RUNS}, window={WINDOW}, stride={STRIDE}) ...")
    X, y, groups = load_tep("training", max_runs=MAX_RUNS)
    print(f"X shape: {X.shape}  (samples, window, features)")
    print(f"y shape: {y.shape}  | classes: {np.unique(y).size} (0=fault-free, 1-20=faults)")
    print(f"NaNs in X: {int(np.isnan(X).sum())}")

    counts = pd.Series(y).value_counts().sort_index()
    counts.index.name = "faultNumber"; counts.name = "n_windows"
    df = counts.reset_index()
    df["pct"] = (100 * df["n_windows"] / df["n_windows"].sum()).round(2)
    df.to_csv(RESULTS / "tep_class_distribution.csv", index=False)
    print("\nClass distribution:\n", df.to_string(index=False))
    imb = counts.max() / counts.min()
    print(f"\nImbalance ratio (max/min class): {imb:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.bar(counts.index, counts.values)
        plt.xlabel("Fault number (0 = normal)"); plt.ylabel("# windows")
        plt.title(f"TEP class distribution (max_runs={MAX_RUNS})")
        plt.tight_layout(); plt.savefig(RESULTS / "tep_class_distribution.png", dpi=120)
        print(f"\nSaved plot -> {RESULTS/'tep_class_distribution.png'}")
    except Exception as e:
        print("Plot skipped:", e)

    print("\nSTEP 01 done. Next: step02 (split + standardize + cache to processed/).")

if __name__ == "__main__":
    main()
