"""STEP 02 — Preprocess TEP with the CANONICAL protocol (fixes flaw #7).

Run:  python src/step02_preprocess_tep.py

- TRAIN/VAL come from the *_Training simulations (grouped by run -> leakage-free).
- TEST comes from the separate *_Testing simulations (fault onset at sample 160),
  loaded IN ORDER with metadata (fault, run, start-sample) so detection delay is
  computable. This is the standard Rieth/Downs-Vogel evaluation protocol.
- Standardization fit on TRAIN only.

Outputs (data/tep_extended/processed/):
  X_train y_train  X_val y_val  X_test y_test  scaler_mean scaler_std
  test_meta.csv (fault, run, start)   manifest.json
"""
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from config import DATA, WINDOW, STRIDE, SEED
from tep_loader import load_tep, load_tep_ordered

TRAIN_RUNS = 100    # simulations per fault from *_Training
TEST_RUNS = 50      # simulations per fault from *_Testing
OUT = DATA / "tep_extended" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print(f"TRAIN/VAL from *_Training (max_runs={TRAIN_RUNS}) ...")
    Xtr_all, ytr_all, groups = load_tep("training", max_runs=TRAIN_RUNS)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr_idx, val_idx = next(gss.split(Xtr_all, ytr_all, groups))
    Xtr, ytr = Xtr_all[tr_idx], ytr_all[tr_idx]
    Xval, yval = Xtr_all[val_idx], ytr_all[val_idx]

    print(f"TEST from *_Testing (max_runs={TEST_RUNS}, ordered) ...")
    Xte, yte, meta = load_tep_ordered("testing", max_runs=TEST_RUNS)

    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    norm = lambda a: ((a - mean) / std).astype(np.float32)
    Xtr, Xval, Xte = norm(Xtr), norm(Xval), norm(Xte)

    np.save(OUT / "X_train.npy", Xtr); np.save(OUT / "y_train.npy", ytr)
    np.save(OUT / "X_val.npy", Xval); np.save(OUT / "y_val.npy", yval)
    np.save(OUT / "X_test.npy", Xte); np.save(OUT / "y_test.npy", yte)
    np.save(OUT / "scaler_mean.npy", mean); np.save(OUT / "scaler_std.npy", std)
    meta.to_csv(OUT / "test_meta.csv", index=False)

    manifest = {"protocol": "canonical (train=Training, test=Testing)",
                "train_runs": TRAIN_RUNS, "test_runs": TEST_RUNS,
                "window": WINDOW, "stride": STRIDE, "seed": SEED,
                "n_features": int(Xtr.shape[-1]), "n_classes": int(np.unique(ytr).size),
                "n_train": int(len(Xtr)), "n_val": int(len(Xval)), "n_test": int(len(Xte)),
                "test_onset_sample": 160, "sample_minutes": 3}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("STEP 02 done (canonical protocol).")


if __name__ == "__main__":
    main()
