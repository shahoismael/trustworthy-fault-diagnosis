# Figures

Each figure lists its caption, description, source, and exact location in the manuscript. Figures 1–2 are conceptual (to be drawn); Figures 3–6 are produced from the simulation outputs in `results/`.

---

## Figure 1 — Why overconfidence is the core problem
**Location:** Section 1 (Introduction), opening motivation figure.
**Caption.** "A confident classifier is not a trustworthy one: as an input drifts away from the training distribution, a softmax model keeps reporting high confidence for the wrong class, whereas an uncertainty-aware model raises its uncertainty."
**Description.** Two side-by-side panels. Left: softmax confidence stays high and wrong as a signal is progressively perturbed. Right: an uncertainty-aware score rises as the input leaves the training manifold. Reproduce the *concept* of Sensoy et al. (2018, Fig. 1) on Tennessee Eastman windows (a normal window morphing into an unknown fault) — do **not** reprint the original image (copyright). Own result, our data.
**Source.** To be generated (concept adapted; data = TEP unknown-fault windows).

---

## Figure 2 — Unified framework architecture
**Location:** Section 3.3 (Methods).
**Caption.** "One backbone, four heads: the shared 1-D CNN feeds classification (focal loss), calibrated detection (validation-set threshold), open-set rejection (feature-space Mahalanobis), and gradient attribution."
**Description.** Block diagram: windowed input (W×d) → Conv-BN-ReLU-Pool → Conv-ReLU-GAP → 128-d features → (i) softmax head; (ii) fault score s=1−p_normal with threshold τ; (iii) Mahalanobis on features; (iv) input-gradient saliency. Emphasize that every downstream task reads from the same features.
**Source.** To be drawn.

---

## Figure 3 — Tennessee Eastman class balance and baseline comparison
**Location:** Section 4.1 (Results).
**Caption.** "Class distribution of the windowed Tennessee Eastman data and macro-F1 of baseline models."
**Description.** (a) Bar chart of window counts per fault class; (b) bar chart of baseline macro-F1 (PCA-NC, SVM, RF, MLP, CNN).
**Source.** `results/tep_class_distribution.png`, `results/baselines_macroF1.png`.

---

## Figure 4 — Variable attribution
**Location:** Section 4.6 (Results).
**Caption.** "Input-gradient saliency ranks the reactor cooling-water temperature and condenser valve as the most influential variables, consistent with the physical fault mechanisms."
**Description.** Horizontal bar chart of top-8 per-variable saliency (xmeas_21 highest), annotated with physical meaning.
**Source.** `results/unified_interpretability.csv` (generate bar chart).

---

## Figure 5 — Calibrated operating point
**Location:** Section 4.3 (Results).
**Caption.** "Fixing the decision threshold on validation normal windows yields FDR 0.90 at FAR 0.06; a naïve argmax rule instead produces a false-alarm rate near 0.9."
**Description.** Detection-rate versus false-alarm-rate curve with the calibrated operating point marked at FAR = 0.06, and the naïve argmax point marked for contrast.
**Source / files.** Bar summary `final_figures/fig5_operating_point.png` (from `results/unified_detection.csv`); full curve `final_figures/fig5_curve.png` (from `results/arr_fault_score.csv`, produced by `build_all_figures.py`).

---

## Figure 6 — Open-set score separability
**Location:** Section 4.4 (Results).
**Caption.** "Feature-space Mahalanobis distance separates unknown faults (16/17/18) from known classes (AUROC 0.82), while softmax-based scores stay near chance."
**Description.** Overlaid score distributions (known vs unknown) for the Mahalanobis score and, for contrast, the energy/entropy scores; ROC curves inset.
**Source / files.** Bar summary `final_figures/fig6_openset_auroc.png` (from `results/unified_openset.csv`); distributions + ROC `final_figures/fig6_distribution.png` (from `results/arr_scores.csv`, produced by `build_all_figures.py`).
