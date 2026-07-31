# AI-Driven Fault Diagnosis and Soft Sensing in Petrochemical Processes
### Research Gap Analysis, Novelty Statement, and Proposal
*Prepared July 2026 · Focus: the intersection of machine learning (classification + optimization) and chemical/petrochemical process engineering · Open-access datasets prioritized*

---

## 1. Executive summary

The intersection of machine learning and chemical/petrochemical process engineering is mature enough to be competitive but still has clearly exploitable gaps. Thousands of papers now report high accuracy on the **Tennessee Eastman Process (TEP)** benchmark and on refinery **soft-sensor** datasets, yet almost all of them share the same four weaknesses: they evaluate on *balanced, simulated* data, they treat the model as a *black box*, they *ignore rare and unknown faults*, and they *do not transfer* across operating modes or from simulation to a real plant.

This document proposes a single, feasible, and novel research thread that attacks those weaknesses head-on:

> **A physics-informed, uncertainty-aware framework for cross-domain fault diagnosis and soft sensing in petrochemical processes — trained on simulated data (TEP), validated on real pilot-plant data (Cranfield three-phase flow facility), with calibrated rejection of rare/unknown faults and interpretable root-cause attribution.**

Everything proposed uses **open-access, highly-used datasets** and is achievable at the scale of a single MSc/PhD project with no proprietary plant access.

---

## 2. Scope: which "mix" this targets

You asked to mix AI with chemical/petrochemical engineering rather than chase every keyword. The strongest, most fundable overlap concentrates two of your keywords — **classification** (fault diagnosis) and **optimization** (setpoint/quality optimization and soft sensing) — onto process-industry data. The four research pillars below define the current state of the art; Section 4 extracts the gaps.

---

## 3. State of the art (four pillars)

### Pillar 1 — Fault detection & diagnosis (classification)
Deep learning on the TEP benchmark is a crowded, well-cited field. Deep CNNs (Wu et al., 2018, ~465 citations) [7], extended deep belief networks (Wang et al., 2019, ~359 citations) [3], process-topology CNNs that inject connectivity priors for interpretability (Wu et al., 2021) [5], attention-LSTM-FCN hybrids (Xiong et al., 2022) [10], temporal CNN/GAN augmentation (Lomov et al., 2021, ~103 citations) [1], and multiscale wavelet-entropy + signed-directed-graph root-cause methods (Ali et al., 2023) [9] all report strong fault-classification accuracy. Two recent papers explicitly flag the field's blind spots: transfer learning is needed when the control strategy changes (Souza et al., 2023) [4], and ensemble/temporal preprocessing is still being tuned specifically to TEP dynamics (Hou, 2026) [6].

### Pillar 2 — Soft sensing / quality prediction (regression)
Soft sensors estimate hard-to-measure quality variables (sulfur content, boiling points, C2/C5 in LPG) from cheap process measurements. The canonical survey is Sun et al., 2021 (~480 citations) [3-ss]. Representative work: spatiotemporal-attention LSTM on an industrial hydrocracker (Yuan et al., 2020, ~385 citations) [7-ss]; graph-convolutional soft sensors that expose variable interactions (Jia et al., 2023, ~129 citations) [6-ss]; deep soft sensing of LPG de-ethanization/debutanization in a real refinery (Sifakis et al., 2023) [2-ss]; and the foundational PLS refinery soft sensor (Wang et al., 2010, ~186 citations) [10-ss]. A 2025 review confirms deep learning is now the primary direction for the field (Gallareta et al., 2025) [5-ss].

### Pillar 3 — Reinforcement learning & optimization for process control
RL is moving from games to plants. Reviews (Devarakonda et al., 2025 [1-rl]; Rajasekhar et al., 2025 [10-rl]; Szatmári et al., 2025 [9-rl]) converge on the same open problems: sample efficiency, safety constraints, generalization, and the **sim-to-real gap**. Notable applied results: apprenticeship + inverse RL to warm-start from historical data (Mowbray et al., 2021) [2-rl]; control-informed RL that embeds PID structure for robustness (Bloor et al., 2024) [5-rl]; RL cutting steam/time in half on a *real* distillation plant (Kubosawa et al., 2021–2022) [3-rl][8-rl]; and a 24.9% cost reduction on a solvent-switch pilot at J&J (Elmaz et al., 2023) [4-rl].

### Pillar 4 — Physics-informed / hybrid modeling
The fastest-growing sub-field fuses first-principles equations with neural networks to fight data scarcity and physical inconsistency. Physics-informed RNNs improve generalization and shrink the required data (Asrav et al., 2023) [1-pi]; physics-guided initialization and loss terms speed convergence on CSTR data (Gallup et al., 2023) [2-pi]; universal differential equations recover hidden kinetics in fermentation (Bangi et al., 2022) [3-pi]; and process-structure-constrained RNNs improve MPC prediction (Wu et al., 2020) [6-pi]. Critically, PINNs are shown to handle **plant–model mismatch** better than pure data-driven models (Moayedi et al., 2024) [7-pi] — the exact problem that kills naive deployments.

---

## 4. Research gaps (the opening)

| # | Gap | Evidence it's still open | Why it matters |
|---|-----|--------------------------|----------------|
| G1 | **Simulated-only validation.** ~90% of TEP fault-diagnosis papers never touch real process data. | Souza et al. flag data-distribution shift as unaddressed [4]; Cranfield is described as uniquely offering *real* noise vs. TEP. | Accuracy on clean simulations does not survive real, noisy, multiphase data. |
| G2 | **Class imbalance & rare faults.** Papers report macro-accuracy on balanced fault sets; rare/incipient faults are under-detected. | Standard TEP splits are balanced by construction; Rieth 2017 offers 500 runs but studies rarely exploit tail behavior. | In real plants, dangerous faults are rare — precisely the ones models miss. |
| G3 | **Open-set / unknown faults.** Models assume a closed set of known fault classes and cannot say "this is a fault I've never seen." | No closed-set TEP paper rejects out-of-distribution faults; all classify into known labels. | A misclassified novel fault is worse than a flagged unknown. |
| G4 | **No calibrated uncertainty.** Predictions are point estimates with no trustworthy confidence. | Uncertainty quantification largely absent across Pillars 1–2; only nascent in RL safety work [1-rl]. | Operators need to know when *not* to trust the model. |
| G5 | **Black-box interpretability.** Root-cause attribution is rare; PTCN [5] and SDG [9] are exceptions, not the norm. | Most CNN/LSTM papers report accuracy only, no causal/variable attribution. | Unexplained alarms are ignored on the plant floor. |
| G6 | **Cross-domain / sim-to-real transfer.** Almost no work trains on TEP and tests on a different (real) facility. | Sim-to-real named as an open problem in every RL review [1-rl][10-rl]; unaddressed for classification. | Reusing models across units/plants is the industrial prize. |
| G7 | **Fragmented tasks.** Fault diagnosis, soft sensing, and optimization are studied in isolation. | No single open framework does classification + regression + physics constraints on shared data. | Real operations need all three coupled. |

**The concentrated opportunity:** no published work combines **calibrated uncertainty + open-set rejection + class-imbalance handling + interpretability + simulation→real transfer** in one framework for petrochemical fault diagnosis. That combination is the novelty.

---

## 5. Novelty statement

> This research proposes **PICUP-FDD** (*Physics-Informed, Calibrated, Uncertainty-aware, Open-set framework for Fault Diagnosis and soft sensing*): the first framework to (i) train on the simulated TEP benchmark and validate on the *real* Cranfield multiphase-flow facility, (ii) reject unknown faults via calibrated evidential uncertainty, (iii) remain robust to severe class imbalance and incipient faults, and (iv) deliver process-topology-aware, interpretable root-cause attribution — all on fully open datasets.

Each ingredient exists somewhere in the literature; **none has been combined**, and the simulation→real petrochemical validation is essentially unclaimed.

---

## 6. Proposed research directions

### 6A — Primary project (classification-centric, highest novelty)
**Title:** *Cross-domain, uncertainty-aware fault diagnosis for petrochemical processes with interpretable root-cause attribution.*

**Core idea.** Build a temporal backbone (attention-based Temporal CNN or LSTM-FCN, following the best TEP architectures [1][10]) and layer on four contributions mapped directly to the gaps:

1. **Class-imbalance robustness (G2):** class-balanced / focal loss + controlled GAN or SMOTE-style augmentation for rare faults (extends the GAN-augmentation idea of Lomov et al. [1]).
2. **Calibrated uncertainty + open-set rejection (G3, G4):** evidential deep learning or Monte-Carlo dropout to produce per-prediction confidence and an "unknown fault" reject option; evaluate calibration with ECE and open-set metrics (AUROC on held-out faults).
3. **Domain adaptation, sim→real (G1, G6):** domain-adversarial training (DANN) or CORAL to transfer a TEP-trained model to the Cranfield real dataset; report the accuracy drop with/without adaptation.
4. **Interpretability (G5):** attention maps + SHAP compared against a process-topology / signed-directed-graph prior (as in PTCN [5] and Ali et al. [9]) to attribute faults to specific variables/units.

**Why feasible:** all four are established techniques; the novelty is their *composition* and the *sim→real petrochemical benchmark*, not inventing a new optimizer.

### 6B — Alternative / complementary project (optimization + regression)
**Title:** *Physics-informed, uncertainty-aware soft sensor with Bayesian setpoint optimization for distillation quality.*

Train a hybrid PINN soft sensor (universal-ODE style [3-pi], plant-mismatch-robust [7-pi]) on the **debutanizer** and **SRU** benchmark datasets to predict product quality with confidence intervals, then use **Bayesian optimization** (or safe RL [5-rl]) over the calibrated model to recommend energy-minimizing setpoints. This directly hits your "optimization" keyword and pairs naturally with 6A as a second paper.

**Recommendation:** Lead with **6A** (cleaner novelty, strongest open datasets, single coherent story). Use **6B** as the follow-on paper or thesis chapter 2.

---

## 7. Open-access datasets (ranked, verified)

### Tier 1 — Core petrochemical benchmarks (use these)

| Dataset | What it is | Task | Access | Notes |
|---------|-----------|------|--------|-------|
| **Extended Tennessee Eastman Process** (Rieth, Amsel, Tran & Cook, 2017) | 20 fault classes + normal, 500 simulation runs each, 3-min sampling | Fault detection & diagnosis (classification) | **Open** — Harvard Dataverse, DOI `10.7910/DVN/6C3JR1`; CSV mirror on Kaggle | The field's default benchmark; large enough for deep learning and for rare-fault/imbalance studies. |
| **Cranfield Three-Phase / Multiphase Flow Facility** (Ruiz-Cárcel et al., 2015) | Real pilot-scale air–water–oil rig, 6 seeded faults, real process noise | Fault detection + soft sensing | **Open** — IEEE DataPort | *Real* data — the key to the sim→real novelty (G1/G6). Direct petrochemical/upstream relevance. |
| **Fortuna et al. Soft-Sensor Benchmark: Debutanizer column + SRU** | Refinery debutanizer (7 vars, ~2400 samples) and Sulphur Recovery Unit (~10,000 samples) | Soft sensing / quality regression | **Open** — distributed with the Fortuna et al. book; widely mirrored | The classic petrochemical soft-sensor benchmark for Project 6B. |

### Tier 2 — Auxiliary datasets (method development, imbalance, drift)

| Dataset | Use in this project |
|---------|--------------------|
| **SECOM** (UCI) | Highly imbalanced manufacturing classification — pretest imbalance methods (G2) before TEP. |
| **Gas Sensor Array Drift** (UCI, Vergara et al., ~10k samples) | Concept-drift + chemical sensing — validate domain-adaptation machinery (G6). |
| **Combined Cycle Power Plant** (UCI, Tüfekci 2014) | Well-cited energy regression — sanity-check the soft-sensor/optimization pipeline (6B). |
| **Steel Plates Faults** (UCI) | Small multi-class fault classification — fast prototyping of the classifier + interpretability. |

All Tier-1 datasets are free, downloadable without institutional access, and already carry strong citation histories in the FDD/soft-sensor literature — satisfying the "high-cited, applicable, open-access" requirement.

---

## 8. Methodology & experimental plan

1. **Data pipeline.** Standardize TEP (Rieth CSV), Cranfield, and the soft-sensor sets into a common windowed-time-series format; document normalization and train/val/test splits with non-overlapping simulation seeds to avoid leakage.
2. **Baselines (must reproduce first).** PCA/PLS (classical) → 1D-CNN → LSTM → attention-LSTM-FCN. Reproducing Wu 2018 [7] and Xiong 2022 [10] establishes credibility and a fair comparison.
3. **Add contributions incrementally** (ablation-friendly): imbalance loss → uncertainty head → open-set rejection → domain adaptation → interpretability. Each addition is a table row and a claim.
4. **Sim→real protocol.** Train on TEP, adapt to Cranfield with a small labeled fraction, report degradation curves vs. labeled-fraction.
5. **Interpretability validation.** Check whether attributed variables match the known process topology / fault mechanism — this is the qualitative evidence reviewers want.

---

## 9. Evaluation metrics & baselines

- **Classification:** Fault Detection Rate, False Alarm Rate, per-class recall (emphasize rare faults), detection delay, macro-F1.
- **Uncertainty:** Expected Calibration Error (ECE), reliability diagrams.
- **Open-set:** AUROC / AUPR on held-out unknown faults, open-set classification rate.
- **Transfer:** accuracy gap TEP→Cranfield with/without adaptation vs. labeled-target fraction.
- **Soft sensor (6B):** RMSE, R², plus interval coverage (PICP) for the uncertainty.
- **Baselines:** PCA, PLS, SVM, plain CNN/LSTM, and the best published attention model — reproduced, not just cited.

---

## 10. Feasibility, risks, and mitigation

| Risk | Mitigation |
|------|-----------|
| Cranfield labels/faults differ from TEP's 20 classes | Frame sim→real as *detection* + *open-set* transfer, not identical-label classification; this is scientifically stronger anyway. |
| Domain adaptation underperforms | Fall back to few-shot fine-tuning; report the honest degradation curve (still a publishable result). |
| Scope creep across 6A + 6B | Ship 6A as paper 1; treat 6B as a separate chapter/paper. |
| Reproducing baselines is slow | Use published TEP code/repos (e.g., open FDD benchmark suites) as a starting point. |

**Compute:** single-GPU scale. **Data:** 100% open. **Timeline:** ~3 months baselines + core contributions, ~2 months transfer + interpretability, ~1 month writing.

---

## 11. Expected contributions & target venues

**Contributions:** (1) first open sim→real petrochemical fault-diagnosis benchmark protocol; (2) a composed framework delivering calibrated uncertainty + open-set rejection + imbalance robustness + interpretability; (3) an ablation quantifying what each component buys.

**Target journals:** *Computers & Chemical Engineering*, *Journal of Process Control*, *Process Safety and Environmental Protection*, *ISA Transactions*, *IEEE Transactions on Industrial Informatics* — all of which actively publish exactly this work (see references).

---

## 12. A candid counter-view (steel-man against this proposal)

Before committing, weigh the strongest objections:

- **"TEP is saturated."** Largely true for *closed-set accuracy*. The defense: this proposal deliberately leaves that saturated axis and competes on transfer, calibration, and open-set — where the leaderboard is nearly empty.
- **"Combining five known techniques isn't novel."** A fair challenge. Novelty here is empirical and integrative, not algorithmic. Strengthen it by making the **sim→real benchmark protocol** itself a contribution (a reusable, citable artifact), not just a results table.
- **"Cranfield ≠ a real refinery."** Correct — it is a pilot rig, not a full petrochemical plant. Claim "toward sim-to-real," not "solved sim-to-real," and be explicit about the gap.
- **Publication-bias caution:** the cited accuracy numbers are self-reported on favorable splits. Treat every headline accuracy as an upper bound and reproduce before comparing.

If those objections worry you more than they reassure, 6B (physics-informed soft sensing + optimization) is the lower-risk alternative with clearer engineering payoff.

---

## References

**Fault detection & diagnosis (Pillar 1)**
[1] Lomov et al. (2021). *Fault detection in TEP with temporal deep learning models.* J. Ind. Inf. Integr. https://consensus.app/papers/details/d9fe2b186f6950f2a6dc29ca567ad34f/
[3] Wang et al. (2019). *Deep learning fault diagnosis with extended deep belief network.* ISA Transactions. https://consensus.app/papers/details/792d32468345532b9e18f545065eaa98/
[4] Souza et al. (2023). *CNNs and transfer learning for FDD.* Evolving Systems. https://consensus.app/papers/details/2c477ed7f0c2580082c978e48ab0b4ee/
[5] Wu et al. (2021). *Process topology convolutional network for fault diagnosis.* Process Saf. Environ. Prot. https://consensus.app/papers/details/1bae81c4757d5a089ae5a22fc40f5f29/
[6] Hou (2026). *Temporal preprocessing + ensemble learning on TEP (ChemFaultNet).* CNML. https://consensus.app/papers/details/ec56361b265152c3abe883c63203952f/
[7] Wu et al. (2018). *Deep CNN for chemical process fault diagnosis.* Comput. Chem. Eng. https://consensus.app/papers/details/1d7ac7502c105dcc9a8b3c67064a3167/
[9] Ali et al. (2023). *Wavelet-entropy ML for multiscale process monitoring.* Process Saf. Environ. Prot. https://consensus.app/papers/details/72f6d0f2ffc055e9a4381f51f34cf59e/
[10] Xiong et al. (2022). *Attention-based LSTM-FCN for chemical process fault diagnosis.* Chin. J. Chem. Eng. https://consensus.app/papers/details/7989e84cc39954db8f6e788b37fb6aaa/

**Soft sensing (Pillar 2)**
[3-ss] Sun et al. (2021). *A survey on deep learning for data-driven soft sensors.* IEEE Trans. Ind. Inf. https://consensus.app/papers/details/caea39ccae21525db05cce87dce263c1/
[7-ss] Yuan et al. (2020). *Spatiotemporal attention LSTM soft sensor.* IEEE Trans. Ind. Electron. https://consensus.app/papers/details/b401aa10b09e5daabe97f90edf25aaf7/
[6-ss] Jia et al. (2023). *Graph convolutional network soft sensor.* J. Process Control. https://consensus.app/papers/details/b09fae37c44858f6b4795d66771a8d91/
[2-ss] Sifakis et al. (2023). *Soft sensing of LPG processes using deep learning.* Sensors. https://consensus.app/papers/details/715244bd267f581bb82aa32aac2b3939/
[5-ss] Gallareta et al. (2025). *Advancements in soft-sensor technologies (review).* IEEE Sensors J. https://consensus.app/papers/details/255ca92e10e35cf68af8a631129b63b9/
[10-ss] Wang et al. (2010). *PLS data-driven soft sensor for a refining process.* IEEE Trans. Ind. Inf. https://consensus.app/papers/details/808c3b9222e35c2cae30f4a32d442d72/

**Reinforcement learning & optimization (Pillar 3)**
[1-rl] Devarakonda et al. (2025). *Recent advances in RL for chemical process control.* Processes. https://consensus.app/papers/details/51a8dd635b4d54ba933841e62f460c9c/
[2-rl] Mowbray et al. (2021). *Apprenticeship + RL for optimal control policy.* AIChE J. https://consensus.app/papers/details/04a86271ea305b46a128e7fd844a987a/
[3-rl] Kubosawa et al. (2021). *Computing operation procedures via deep RL on a distillation plant.* Control Eng. Pract. https://consensus.app/papers/details/02fe76fdd8175d90bc189eaa40886d95/
[4-rl] Elmaz et al. (2023). *RL for solvent-switch optimization.* Comput. Chem. Eng. https://consensus.app/papers/details/2a1e719b5b3d57c1a401b0469c18d294/
[5-rl] Bloor et al. (2024). *Control-informed RL for chemical processes.* Ind. Eng. Chem. Res. https://consensus.app/papers/details/e565c5bcac8b5e29a81e61672160fda1/
[8-rl] Kubosawa et al. (2022). *Sim-to-real transfer in RL for chemical plants.* SICE JCMSI. https://consensus.app/papers/details/a284fbfabd6b571ba952e4ce93a6de95/
[9-rl] Szatmári et al. (2025). *RL in hierarchical chemical process control (CRISP-RL).* Chem. Eng. J. Adv. https://consensus.app/papers/details/fb0e7a67bfe2535e92eeb84a7c460daf/
[10-rl] Rajasekhar et al. (2025). *RL in process control: a comprehensive survey.* Int. J. Syst. Sci. https://consensus.app/papers/details/34ca6b488b9451f48a85da4189961c9a/

**Physics-informed / hybrid modeling (Pillar 4)**
[1-pi] Asrav et al. (2023). *Physics-informed RNNs for dynamic process systems.* Comput. Chem. Eng. https://consensus.app/papers/details/6b93418e62e45882a280a443b536de40/
[2-pi] Gallup et al. (2023). *Physics-guided NNs with domain knowledge for hybrid modeling.* Comput. Chem. Eng. https://consensus.app/papers/details/6b0238691fe85dc19bbd6bdc75e3b152/
[3-pi] Bangi et al. (2022). *PINN hybrid modeling of batch fermentation (UDEs).* Chem. Eng. Res. Des. https://consensus.app/papers/details/c0fa8ef9a69f5c41b727728d424051ce/
[6-pi] Wu et al. (2020). *Process-structure RNN modeling for MPC.* J. Process Control. https://consensus.app/papers/details/c15ed45c7ec8570ba7d9ee98d07de5ea/
[7-pi] Moayedi et al. (2024). *PINNs for process systems: handling plant–model mismatch.* Ind. Eng. Chem. Res. https://consensus.app/papers/details/e561c78fe0be55eab89a14dc8cfc3d03/

**Datasets**
- Rieth, Amsel, Tran & Cook (2017). *Additional TEP simulation data for anomaly detection evaluation.* Harvard Dataverse, DOI 10.7910/DVN/6C3JR1.
- Ruiz-Cárcel et al. (2015). *Statistical process monitoring of a multiphase flow facility.* Control Eng. Pract.; dataset on IEEE DataPort (Three-Phase Flow Facility).
- Fortuna, Graziani, Rizzo & Xibilia (2007). *Soft Sensors for Monitoring and Control of Industrial Processes* — Debutanizer & SRU benchmark datasets.

*Citation counts are as reported by the indexing service at retrieval (July 2026) and should be treated as approximate. Reproduce baseline results before making comparative claims.*
