"""STEP 09 — Unified CNN framework (fixes the 'different model per pillar' flaw).

Run:  python src/step09_unified_cnn.py   (needs torch + step02 cache)

ONE CNN backbone provides ALL pillars, so PICUP-FDD is a single system:
  - Classification + incipient-fault recovery : focal-loss CNN, macro-F1 + rare recall
  - Open-set recognition : four scores from the SAME model
        MSP (max softmax prob), Energy (Liu 2020), Entropy,
        Mahalanobis on penultimate features (Lee 2018)   -> AUROC vs unknowns
  - Interpretability : input-gradient saliency of the SAME model -> per-variable importance

Trained on KNOWN faults only (UNKNOWN_FAULTS held out) so open-set is honest.

Outputs (results/):
  unified_metrics.csv          closed-set macro-F1 + rare-fault recall
  unified_openset.csv          AUROC for MSP / energy / entropy / mahalanobis
  unified_interpretability.csv top variables (gradient saliency)
"""
import numpy as np, pandas as pd
from config import DATA, RESULTS, SEED, RARE_FAULTS, UNKNOWN_FAULTS, TEP_FEATURES
try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch not installed. pip install torch")
from sklearn.metrics import f1_score, recall_score, roc_auc_score

PROC = DATA / "tep_extended" / "processed"
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BS, LR = 15, 128, 1e-3

def load():
    g = lambda n: np.load(PROC / n)
    return (g("X_train.npy"), g("y_train.npy"), g("X_val.npy"), g("y_val.npy"),
            g("X_test.npy"), g("y_test.npy"))

class CNN(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(f, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3))
        self.head = nn.Linear(128, c)
    def features(self, x): return self.body(x.transpose(1, 2))
    def forward(self, x):  return self.head(self.features(x))

def focal(logits, target, gamma=2.0, w=None):
    ce = F.cross_entropy(logits, target, weight=w, reduction="none")
    return ((1 - torch.exp(-ce)) ** gamma * ce).mean()

def main():
    Xtr, ytr, Xval, yval, Xte, yte = load()
    known = ~np.isin(ytr, UNKNOWN_FAULTS)
    kcl = np.unique(ytr[known]); kmap = {c: i for i, c in enumerate(kcl)}
    enc = lambda y: np.array([kmap[v] for v in y])
    C = len(kcl)
    cnt = np.bincount(enc(ytr[known]), minlength=C).astype(float)
    # sqrt-balanced weights: lift minority faults WITHOUT crushing the majority
    # (normal) class -> keeps false-alarm rate low. Plain inverse-freq gave FAR~0.96.
    w = torch.tensor(np.sqrt(cnt.sum() / (C * np.clip(cnt, 1, None))), dtype=torch.float32, device=DEV)

    dl = DataLoader(TensorDataset(torch.tensor(Xtr[known]), torch.tensor(enc(ytr[known]), dtype=torch.long)),
                    batch_size=BS, shuffle=True)
    net = CNN(Xtr.shape[-1], C).to(DEV); opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(EPOCHS):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(); focal(net(xb), yb, w=w).backward(); opt.step()

    # ---- inference on test ----
    net.eval()
    def batched(fn, X, bs=512):
        out = []
        for i in range(0, len(X), bs):
            out.append(fn(torch.tensor(X[i:i+bs]).to(DEV)))
        return torch.cat(out)
    with torch.no_grad():
        logit = batched(lambda t: net(t).cpu(), Xte)
        logit_val = batched(lambda t: net(t).cpu(), Xval)
        feat_te = batched(lambda t: net.features(t).cpu(), Xte).numpy()
        feat_tr = batched(lambda t: net.features(t).cpu(), Xtr[known]).numpy()
    P = F.softmax(logit, 1).numpy()
    Pval = F.softmax(logit_val, 1).numpy()
    c0 = kmap[0]                                   # index of the 'normal' class
    fault_score = 1.0 - P[:, c0]                   # P(any fault) for detection
    fault_score_val = 1.0 - Pval[:, c0]

    # ---- closed-set metrics (known-fault test only) ----
    known_te = ~np.isin(yte, UNKNOWN_FAULTS)
    pred = kcl[P.argmax(1)]
    rows = [{"model": "Unified-CNN(focal)",
             "closed_macro_f1": round(f1_score(yte[known_te], pred[known_te], average="macro", zero_division=0), 4)}]
    for fnum in RARE_FAULTS:
        rows[0][f"recall_fault{fnum}"] = round(recall_score(yte == fnum, pred == fnum, zero_division=0), 4)
    pd.DataFrame(rows).to_csv(RESULTS / "unified_metrics.csv", index=False)
    print("Closed-set:", rows[0])

    # ---- FDR / FAR / detection delay at a CALIBRATED operating point (fix #1) ----
    # Threshold chosen on VALIDATION normal windows for ~5% false-alarm target,
    # then applied to test. This replaces naive argmax (which gave FAR~0.9).
    known_faults = [c for c in kcl if c != 0]
    thr = float(np.quantile(fault_score_val[yval == 0], 0.95))   # 5% val FAR
    nt = yte == 0
    kf = np.isin(yte, known_faults)
    far = float(np.mean(fault_score[nt] >= thr)) if nt.any() else np.nan
    fdr = float(np.mean(fault_score[kf] >= thr)) if kf.any() else np.nan
    delays = []
    meta_path = PROC / "test_meta.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path); onset = 160
        for f in known_faults:
            sub = meta[meta["fault"] == f]
            for run, idxs in sub.groupby("run").groups.items():
                pos = list(idxs)
                hit = [i for i in pos if fault_score[i] >= thr]
                if hit:
                    delays.append(max(0, int(meta.loc[hit[0], "start"]) - onset))
    delay_min = round(float(np.mean(delays)) * 3, 2) if delays else np.nan
    fdd = pd.DataFrame([{"operating_point": "val-calibrated ~5% FAR",
                         "FDR": round(fdr, 4), "FAR": round(far, 4),
                         "mean_detection_delay_min": delay_min}])
    fdd.to_csv(RESULTS / "unified_detection.csv", index=False)
    print("FDD metrics (calibrated):", fdd.to_dict("records")[0])
    # dump arrays for figure generation (Fig 5 FAR-FDR curve)
    np.save(RESULTS / "arr_fault_score.npy", fault_score)
    np.save(RESULTS / "arr_mask_normal.npy", nt)
    np.save(RESULTS / "arr_mask_knownfault.npy", kf)
    np.save(RESULTS / "arr_thr.npy", np.array([thr]))

    # ---- open-set scores (higher = more 'unknown') ----
    is_unknown = np.isin(yte, UNKNOWN_FAULTS).astype(int)
    msp = -P.max(1)
    energy = (-torch.logsumexp(logit, 1)).numpy()
    ent = -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1)
    # Mahalanobis on penultimate features (shared covariance, per-class means)
    mus = np.stack([feat_tr[enc(ytr[known]) == i].mean(0) for i in range(C)])
    cov = np.cov((feat_tr - mus[enc(ytr[known])]).T) + 1e-6 * np.eye(feat_tr.shape[1])
    prec = np.linalg.pinv(cov)
    def maha(F_):
        d = F_[:, None, :] - mus[None]           # (N, C, D)
        m = np.einsum("ncd,de,nce->nc", d, prec, d)
        return m.min(1)                           # distance to nearest class centroid
    maha_te = maha(feat_te)
    ok = is_unknown.sum() and (is_unknown == 0).sum()
    A = lambda s: round(roc_auc_score(is_unknown, s), 4) if ok else np.nan
    op = pd.DataFrame([{"MSP_AUROC": A(msp), "energy_AUROC": A(energy),
                        "entropy_AUROC": A(ent), "mahalanobis_AUROC": A(maha_te),
                        "unknown_faults": str(UNKNOWN_FAULTS), "device": DEV}])
    op.to_csv(RESULTS / "unified_openset.csv", index=False)
    print(op.to_string(index=False))
    # dump open-set score arrays for figure generation (Fig 6 distributions/ROC)
    np.save(RESULTS / "arr_is_unknown.npy", is_unknown)
    np.save(RESULTS / "arr_maha.npy", maha_te)
    np.save(RESULTS / "arr_energy.npy", energy)
    np.save(RESULTS / "arr_entropy.npy", ent)

    # ---- interpretability: input-gradient saliency of the SAME model ----
    idx = np.random.RandomState(SEED).choice(len(Xte), size=min(1500, len(Xte)), replace=False)
    xb = torch.tensor(Xte[idx], requires_grad=True).to(DEV)
    out = net(xb); sc = out.max(1).values.sum()
    net.zero_grad(); sc.backward()
    sal = xb.grad.abs().mean(dim=(0, 1)).cpu().numpy()   # mean |grad| over samples & time -> per variable
    imp = pd.DataFrame({"variable": TEP_FEATURES, "saliency": sal}).sort_values("saliency", ascending=False)
    imp.to_csv(RESULTS / "unified_interpretability.csv", index=False)
    print("Top-8 variables (CNN saliency):\n", imp.head(8).to_string(index=False))
    print("STEP 09 done.")

if __name__ == "__main__":
    main()
