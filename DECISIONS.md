# PICUP-FDD — Locked Decisions (memory file)

## Title (FINAL — evidence-matched, see TITLE.md)
**Trustworthy Deep Fault Diagnosis for Petrochemical Processes: Calibrated, Open-Set Recognition of Known and Unknown Faults**
> Physics-Informed and Sim-to-Real removed from title (not implemented / negative result). See TITLE.md.

## Framework scope (locked after audit)
- Unified 1D-CNN backbone demonstrated on BOTH simulated (TEP, macro-F1 0.88) and REAL (PRONTO, acc 0.977 / AUROC 0.990) petrochemical time-series FDD — same architecture (step09, step15).
- Steel Plates (tabular classification) + Debutanizer/SRU (regression) = AUXILIARY generalization evidence with task-appropriate models; explicitly NOT claimed as the same CNN.
- IndPenSim = SAMPLE-LEVEL detection (no clean 100-batch id); labelled honestly as sample-level, not batch-level.

## Honest limitations (must appear in the paper)
1. Sim→real weight transfer is infeasible (TEP 52 vars ≠ PRONTO 17); reported as a NEGATIVE result (naive 0.34 / adapted 0.44). The positive real-plant result is same-architecture, target-trained (step15), not transfer.
2. Rare/incipient-fault recall is high-variance across seeds (F3/F9/F15) — inherent difficulty; reported with mean±std.
3. Open-set uses one held-out unknown set {16,17,18}; not rotated over all faults.
4. Evidential head underperformed; uncertainty is delivered via entropy/Mahalanobis (supporting, not headline).

## FIGURE 1 (LOCKED — critical)
**Motivation figure = Ref #68, Fig. 1** (Sensoy et al., Evidential Deep Learning to Quantify Classification Uncertainty, NeurIPS 2018).
- Content: rotated-digit experiment — softmax stays confidently WRONG as input rotates, while the evidential model raises uncertainty.
- Why locked: it is the cleanest single visual proof of the core problem PICUP-FDD attacks (overconfident classifiers cannot say "I don't know").
- Use: opening motivation figure; we reproduce the SAME experiment on TEP faults (not the digit image) so Figure 1 becomes our own result, not a reprint.
- Source file: `references/68-78/68-Evidential Deep Learning to Quantify Classification Uncertainty.pdf`
- Note: do NOT reprint the original image (copyright) — reproduce the concept on our data.

## References (locked)
- 77 total. Dropped #67 (duplicate XAI review). Added 68–78.
- Master matrix: `references/Literature_Matrix_MASTER.csv` — 19 columns, 0 PENDING cells.
- Camp_Tag / RQ_Relevance re-mapped to this project's RQs:
  RQ1 open-set · RQ2 uncertainty · RQ3 imbalance · RQ4 sim→real · RQ5 interpretability.

## Datasets (locked — 5)
1. Extended TEP (primary) · 2. PRONTO multiphase (real plant) · 3. IndPenSim (batch bioprocess)
4. Debutanizer + SRU (soft sensor) · 5. Steel Plates (cross-domain generalization)

## Target journals (locked)
Q1 Elsevier/Springer only: RESS · PSEP · EAAI · J. Intelligent Manufacturing · Neural Computing & Applications.

## Sim→real definition (LOCKED = option a)
**Method-level transfer.** Train detector on TEP normal data → transfer the *method* (not weights) → re-fit on PRONTO normal → evaluate detection on PRONTO real faults. Diagnosis (21-class) stays TEP-only. Claim = "method-level sim→real generalization." No shared feature space required.
Status: LOCKED.

## Environment
conda env `picup` (Python 3.11). Activate glitch on Windows → use `conda run -n picup ...`.
