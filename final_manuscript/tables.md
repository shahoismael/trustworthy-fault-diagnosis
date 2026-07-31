# Tables

All numeric values are the real simulation outputs (see `results/` CSVs). Each table lists its exact location in the manuscript.

---

## Table 1 — Datasets used in the study
**Location:** Section 3.1 (Materials and Methods).

| # | Dataset | Type | Process | Task | Role |
|---|---------|------|---------|------|------|
| 1 | Extended Tennessee Eastman (Rieth et al., 2017) | Simulated, continuous | Chemical plant, 52 vars, 20 faults | Multi-class FDD, open-set | Primary |
| 2 | PRONTO multiphase flow (Stief et al., 2019) | Real, continuous | Air–water–oil pilot, 17 vars, 3 seeded faults | Detection | Real-plant validation |
| 3 | IndPenSim (Goldrick et al., 2015) | Simulated, batch | Fed-batch penicillin | Sample-level detection | Auxiliary |
| 4 | Debutanizer + SRU (Fortuna et al., 2007) | Real, continuous | Refinery units | Soft-sensor regression | Auxiliary |
| 5 | Steel Plates Faults | Tabular | Steel plates | 7-class classification | Cross-domain probe |

---

## Table 2 — Classification on Tennessee Eastman
**Location:** Section 4.1 (Results). Baselines on development split; unified model on canonical test set.

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| PCA + nearest centroid | 0.42 | 0.42 |
| Linear SVM | 0.45 | 0.43 |
| Random forest | 0.60 | 0.63 |
| Multilayer perceptron | 0.69 | 0.71 |
| **Unified CNN (canonical test, 5-seed)** | — | **0.88 ± 0.06** |

Unified CNN macro-F1 over 5 seeds {42, 7, 123, 2024, 2025}: mean 0.882, std 0.055, 95% CI ±0.048 (`results/tep_multiseed_classification.csv`). Rare-fault recall is seed-sensitive: F3 0.73 ± 0.41, F9 0.79 ± 0.16, F15 0.26 ± 0.24 — documented as a limitation (§5.7).

---

## Table 3 — Ablation: training objective (canonical test, controlled)
**Location:** Section 4.2 (Results).

| Configuration | Macro-F1 | Recall F3 | Recall F9 | Recall F15 |
|---------------|----------|-----------|-----------|------------|
| Plain cross-entropy | 0.874 | 1.00 | 0.449 | 0.145 |
| Focal loss | **0.918** | 0.908 | **0.653** | 0.421 |
| Focal + sqrt weights | 0.892 | 0.968 | 0.556 | **0.428** |

---

## Table 4 — Detection operating point and sim-to-real transfer
**Location:** Section 4.3 and Section 4.8 (Results).

| Quantity | Value |
|----------|-------|
| Operating point | Validation-calibrated, target FAR = 0.05 |
| Fault-detection rate (FDR) | 0.90 |
| False-alarm rate (FAR) | 0.06 |
| Mean detection delay | 47.8 min |
| Sim→real, naïve transfer (AUROC) | 0.34 |
| Sim→real, few-shot adapted (AUROC) | 0.44 |

---

## Table 5 — Open-set recognition scores (held-out faults 16/17/18)
**Location:** Section 4.4 (Results).

| Score | AUROC |
|-------|-------|
| Maximum softmax probability | 0.43 |
| Energy | 0.44 |
| Predictive entropy | 0.44 |
| **Feature-space Mahalanobis** | **0.82** |
| Mahalanobis, 5-seed mean ± std | 0.845 ± 0.006 |

---

## Table 6 — Same CNN backbone on the real PRONTO plant
**Location:** Section 4.5 (Results).

| Metric | Value |
|--------|-------|
| Accuracy | 0.977 |
| Macro-F1 | 0.973 |
| AUROC | 0.990 |
| Train / test windows | 1129 / 486 |

---

## Table 7 — Top variables by input-gradient attribution
**Location:** Section 4.6 (Results).

| Rank | Variable | Saliency | Physical meaning |
|------|----------|----------|------------------|
| 1 | xmeas_21 | 0.99 | Reactor cooling-water outlet temperature |
| 2 | xmv_10 | 0.47 | Condenser cooling-water valve |
| 3 | xmeas_18 | 0.29 | Stripper temperature |
| 4 | xmeas_9 | 0.27 | Reactor temperature |
| 5 | xmeas_19 | 0.26 | Stripper steam flow |

---

## Table 8 — Auxiliary datasets, 5-seed mean ± std
**Location:** Section 4.7 (Results).

| Experiment | Metric | Mean ± std |
|------------|--------|-----------|
| Steel Plates | Macro-F1 | 0.786 ± 0.006 |
| Steel Plates | Accuracy | 0.765 ± 0.015 |
| Debutanizer soft sensor | R² | 0.994 ± 0.001 |
| SRU soft sensor (y1) | R² | 0.848 ± 0.005 |
| IndPenSim (sample-level) | AUROC | ≈ 1.000 ± 0.000 |

---

## Table 9 — Comparison with published Tennessee Eastman results
**Location:** Section 4.1 / Section 5 (context). Metrics are not all directly comparable; prior work is closed-set.

| Method | Year | Metric | Value | Note |
|--------|------|--------|-------|------|
| PCA T²/SPE | — | avg FDR | ~0.70 | linear MSPC |
| DBN | 2019 | avg FDR | 0.82 | in Wu & Zhao comparison |
| DCNN (Wu & Zhao) | 2018 | avg FDR | 0.88 | 20 faults |
| Extended DBN (Wang et al.) | 2019 | avg FDR | 0.94 | 19-category |
| PTCN (Wu et al.) | 2021 | ACR | 0.94 | 0.97 excl. faults 9,15 |
| **This work** | 2026 | macro-F1 | 0.88 ± 0.06 | canonical test, 5-seed |
| **This work** | 2026 | FDR @ FAR=0.06 | 0.90 | calibrated operating point |
| **This work** | 2026 | open-set AUROC | 0.82 | not reported by prior TEP work |
