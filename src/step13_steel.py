"""STEP 13 — Steel Plates Faults classification (uses dataset #5).

Run:  python src/step13_steel.py

Cross-domain generalization probe: a 7-class fault classification outside the
petrochemical domain. Confirms the pipeline's classification approach transfers
to another fault-diagnosis setting.

Outputs (results/):  steel_metrics.csv
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from config import STEEL_DIR, RESULTS, SEED

FAULTS = ["Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"]

def main():
    df = pd.read_csv(STEEL_DIR / "steel_plates_faults_original_dataset.csv")
    df.columns = [c.strip() for c in df.columns]
    faults = [c for c in FAULTS if c in df.columns]
    y = df[faults].to_numpy().argmax(1)
    drop = set(faults) | {"id", "Id", "ID"}
    X = df[[c for c in df.columns if c not in drop]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    p = m.predict(Xte)
    row = [{"dataset": "SteelPlates", "n_classes": len(faults), "n": len(df),
            "accuracy": round(accuracy_score(yte, p), 4),
            "macro_f1": round(f1_score(yte, p, average="macro", zero_division=0), 4)}]
    pd.DataFrame(row).to_csv(RESULTS / "steel_metrics.csv", index=False)
    print(row[0]); print("STEP 13 done.")

if __name__ == "__main__":
    main()
