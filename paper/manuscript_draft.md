# Cross-Domain Fault Diagnosis for Petrochemical Processes: A Physics-Informed, Uncertainty-Aware Framework with Open-Set Recognition and Simulation-to-Real Transfer

*Working manuscript draft — Abstract figures and Results/Discussion/Conclusion are marked `[RESULT]` and completed after simulation. Citations use author–year; see References. Final title fixed after results.*

---

## Abstract

Data-driven fault detection and diagnosis (FDD) underpins safe operation in the chemical and petrochemical industries, yet most deep-learning methods reported on benchmark processes share three deployment-limiting weaknesses: they are evaluated on balanced, simulated data; they treat every test sample as one of a fixed set of known faults; and they express no calibrated confidence in their predictions. This paper develops PICUP-FDD, a framework that couples a temporal deep-learning backbone with (i) class-imbalance–robust training for rare and incipient faults, (ii) evidential, calibrated uncertainty that supports rejection of unknown faults through open-set recognition, and (iii) a method-level simulation-to-real transfer protocol in which a detector trained on the simulated Tennessee Eastman Process is re-fitted and evaluated on a real multiphase-flow pilot plant. The framework is assessed on five open-access datasets spanning simulated and real, continuous and batch processes. On the Tennessee Eastman benchmark the model attains `[RESULT: macro-F1 / FDR]`; calibrated uncertainty reduces expected calibration error to `[RESULT: ECE]` and rejects held-out unknown faults at `[RESULT: AUROC]`; and the method-level transfer detects real pilot-plant faults at `[RESULT: detection rate]`. Interpretability analysis attributes fault signatures to process variables consistent with `[RESULT]`. The results indicate `[RESULT: headline claim, fixed after experiments]`.

**Keywords:** fault detection and diagnosis; petrochemical processes; open-set recognition; uncertainty quantification; simulation-to-real transfer; Tennessee Eastman Process.

---

## 1. Introduction

Modern refineries and petrochemical plants are instrumented, tightly controlled, and safety-critical. An abnormal event that is caught early can be contained; the same event missed for minutes can escalate into an unplanned shutdown, product loss, or a hazard to personnel and the environment. Automated fault detection and diagnosis (FDD) therefore sits at the centre of abnormal-situation management, and data-driven FDD in particular has matured rapidly as plants have accumulated large volumes of historized sensor data (Ji and Sun, 2022).

Classical multivariate statistical process monitoring built on principal component analysis and partial least squares established the field and remains a strong, interpretable baseline (Qin, 2003; Ji and Sun, 2022). Over the last decade deep learning has pushed reported accuracy higher: convolutional networks (Wu and Zhao, 2018), deep belief networks (Wang et al., 2019), graph and topology-aware networks (Wu et al., 2021), and temporal architectures with data augmentation (Lomov et al., 2021) all report strong results on the Tennessee Eastman Process (TEP), the community's standard chemical-process benchmark (Downs and Vogel, 1993; Rieth et al., 2017). A parallel line of work on soft sensing predicts hard-to-measure quality variables from routine measurements (Sun and Ge, 2021; Yuan et al., 2020).

Despite this progress, three gaps persist between benchmark performance and plant deployment.

First, **the reported evaluations are almost entirely on balanced, simulated data.** TEP splits are balanced by construction, whereas in an operating plant dangerous faults are rare, and incipient faults with a low signal-to-noise ratio are precisely the ones classifiers miss. Second, **the closed-world assumption is rarely questioned.** A model trained on a fixed catalogue of faults will confidently assign any novel disturbance to one of its known classes; a misclassified unknown fault is more dangerous than one flagged as unknown (Yang et al., 2022; Wang et al., 2024). Third, **predictions carry no trustworthy confidence.** Softmax probabilities are systematically overconfident and are poor proxies for out-of-distribution inputs (Sensoy et al., 2018; Liu et al., 2020), so operators cannot tell when to distrust the model. Cutting across all three is the **simulation-to-real gap**: models are validated on simulators, but distribution shift and non-stationary operating conditions degrade them on real plants (Hinder et al., 2023; Lobo et al., 2023).

This paper addresses these gaps jointly rather than in isolation. We propose **PICUP-FDD**, a framework that:

1. handles severe class imbalance and incipient faults through cost-sensitive training and controlled augmentation;
2. produces calibrated, evidential uncertainty and uses it to reject unknown faults via open-set recognition;
3. transfers at the *method* level from a simulated benchmark (TEP) to a real multiphase-flow pilot plant, so that no shared sensor space is required; and
4. attributes each detection to process variables for interpretable root-cause support.

The contributions are: (a) an integrated FDD framework that unifies imbalance robustness, calibrated uncertainty, open-set rejection, and method-level simulation-to-real transfer, which to our knowledge have not previously been combined for petrochemical FDD; (b) a reproducible evaluation protocol across five open-access datasets spanning simulated/real and continuous/batch processes; and (c) an ablation quantifying the contribution of each component. The remainder of the paper is organized as follows. Section 2 reviews related work; Section 3 describes the datasets and methods; Section 4 reports results; Section 5 discusses implications and limitations; Section 6 concludes.

---

## 2. Related Work

### 2.1 Deep learning for chemical-process fault diagnosis
Data-driven FDD on TEP has been dominated by deep architectures. Convolutional models transform windowed sensor data into image-like tensors for spatial–temporal feature extraction (Wu and Zhao, 2018); extended deep belief networks preserve raw-signal information across layers (Wang et al., 2019); process-topology convolutional networks inject connectivity priors to improve both accuracy and explainability (Wu et al., 2021); and recurrent and temporal-convolutional models with GAN-based augmentation improve detection of the notoriously difficult incipient faults (Lomov et al., 2021). Classical PCA/PLS monitoring remains the reference baseline and a source of interpretable detection statistics (Qin, 2003; Ji and Sun, 2022). These works establish the state of the art on closed-set, balanced, simulated data — the setting this paper deliberately moves beyond.

### 2.2 Class imbalance and incipient faults
Because faults are rare, several studies generate synthetic fault samples with generative adversarial networks or diffusion models to rebalance training data. This paper adopts cost-sensitive objectives and controlled augmentation rather than relying on synthesis alone, and evaluates specifically on the hard incipient faults that motivate the imbalance problem.

### 2.3 Uncertainty quantification and open-set recognition
Softmax confidence is a biased out-of-distribution signal (Liu et al., 2020). Evidential deep learning places a Dirichlet distribution over class probabilities and yields an explicit uncertainty mass in a single forward pass, improving both calibration and novelty detection (Sensoy et al., 2018; Gao et al., 2024). Energy-based scores further separate in- and out-of-distribution inputs (Liu et al., 2020). The related problems of anomaly, novelty, open-set, and out-of-distribution detection have been unified into a single taxonomy (Yang et al., 2022), and rigorous cross-benchmarking shows that feature-magnitude–sensitive scores are the most stable across settings (Wang et al., 2024). PICUP-FDD uses evidential uncertainty as its rejection signal and adopts these benchmarking recommendations as its open-set evaluation protocol.

### 2.4 Transfer learning and the simulation-to-real gap
Domain adaptation for chemical-process FDD has been benchmarked on TEP under changing operating conditions, and non-stationarity (concept drift) is recognized as a safety-critical driver of model degradation (Hinder et al., 2023), with model confidence proposed as an unsupervised drift signal (Lobo et al., 2023). Because a simulated benchmark and a real pilot plant rarely share a sensor space, we adopt *method-level* transfer: the detector methodology, not its weights, is carried from TEP to the real facility and re-fitted on target normal data.

### 2.5 Soft sensing, control, and hybrid modelling
Adjacent to diagnosis, deep soft sensors predict product quality in refinery units (Sun and Ge, 2021; Yuan et al., 2020; Sifakis et al., 2023), reinforcement learning optimizes process operation (Devarakonda et al., 2025), and physics-informed hybrid models fight data scarcity and plant–model mismatch (Bangi et al., 2022). These provide the physics-informed and quality-prediction context for the framework but are not its primary evaluation target.

### 2.6 Summary of the gap
No prior study combines calibrated uncertainty, open-set rejection, class-imbalance robustness, and simulation-to-real transfer within a single petrochemical FDD framework evaluated across simulated and real processes. PICUP-FDD targets exactly this intersection.

---

## 3. Materials and Methods

### 3.1 Datasets
Five open-access datasets are used, chosen to span simulated versus real and continuous versus batch operation (Table 1).

- **Extended Tennessee Eastman Process (primary).** Simulated closed-loop chemical process with 20 fault classes plus normal operation, 500 simulation runs per class, and 52 process variables (Downs and Vogel, 1993; Rieth et al., 2017). Used for closed-set diagnosis, imbalance, uncertainty, and open-set experiments.
- **PRONTO multiphase-flow facility (real plant).** Industrial-scale air–water pilot rig with seeded fault conditions and real process noise (Stief et al., 2019). Used as the target for method-level simulation-to-real transfer.
- **IndPenSim.** Simulated industrial-scale fed-batch penicillin fermentation with fault batches (Goldrick et al., 2015). Provides a batch-bioprocess generalization case.
- **Debutanizer and Sulphur Recovery Unit (SRU).** Real refinery soft-sensor benchmark (Fortuna et al., 2007). Supports the quality-prediction track.
- **Steel Plates Faults.** Multi-class fault dataset used as a cross-domain generalization probe.

*Table 1. Datasets: type, process, task, and role. `[insert table]`*

### 3.2 Data preparation
Each multivariate series is segmented into overlapping windows (length `W`, stride `S`). For TEP, samples before fault onset are discarded from faulty runs so that labels reflect the active fault. Splits are performed **by simulation run** using grouped shuffling, so that no run contributes to more than one of the training, validation, and test partitions; this prevents temporal leakage between windows of the same run. Features are standardized with statistics computed on the training partition only and applied unchanged to validation and test. All preprocessing is deterministic under a fixed random seed and cached for reproducibility.

### 3.3 Backbone
The diagnosis backbone is a temporal deep network operating on standardized windows `X ∈ R^{W×52}`, following the strongest reported TEP architectures — a one-dimensional convolutional feature extractor with an attention-augmented recurrent head (Wu and Zhao, 2018; Yuan et al., 2020). A Transformer variant is included as a modern comparison baseline (Wen et al., 2023).

### 3.4 Class-imbalance robustness
To address rare and incipient faults, training uses a class-balanced/focal objective, complemented by controlled augmentation of minority faults. Performance is reported per class with emphasis on the incipient faults.

### 3.5 Calibrated uncertainty and open-set recognition
The classification head is replaced by an evidential (Dirichlet) layer that yields, in a single forward pass, class beliefs and an explicit uncertainty mass (Sensoy et al., 2018; Gao et al., 2024). At inference, samples whose uncertainty (or energy score; Liu et al., 2020) exceeds a validation-calibrated threshold are rejected as *unknown fault*. A subset of fault classes is held out entirely during training and used only to evaluate open-set rejection, following the cross-benchmarking protocol of Wang et al. (2024).

### 3.6 Method-level simulation-to-real transfer
Because TEP and PRONTO do not share a sensor space, transfer is performed at the method level (locked design decision): the detector is trained on TEP normal data; the *methodology* — architecture, uncertainty head, and thresholding rule — is then re-fitted on PRONTO normal data and evaluated on PRONTO faults. Diagnosis (21-class) remains TEP-only. The reported quantity is the fault-detection performance on real data with and without the calibrated-uncertainty component.

### 3.7 Interpretability
Fault attributions are produced with attention weights and SHAP values and compared against known process topology, so that flagged variables can be checked against the physical fault mechanism (Wu et al., 2021).

### 3.8 Evaluation protocol
Classification is reported with fault-detection rate, false-alarm rate, per-class recall (with emphasis on rare faults), detection delay, and macro-F1. Uncertainty is reported with expected calibration error and reliability diagrams. Open-set is reported with AUROC/AUPR on held-out unknown faults. Transfer is reported as the detection gap on PRONTO with and without adaptation. All deep results are averaged over multiple random seeds with dispersion reported, and baselines (PCA, PLS, SVM, plain CNN/LSTM, Transformer) are reproduced under the same protocol rather than quoted. Code and processed-data pipelines are released for reproducibility.

---

## 4. Results
`[RESULT — completed after simulation: baseline table; imbalance ablation; calibration/ECE; open-set AUROC; sim→real transfer; interpretability case studies.]`

## 5. Discussion
`[RESULT — after experiments: which component drives the headline claim; comparison to prior TEP results; deployment and real-time considerations; threats to validity.]`

## 6. Conclusion
`[RESULT — after experiments.]`

---

## Declarations
- **Data availability:** All five datasets are open access (see Section 3.1 for sources).
- **Code availability:** Preprocessing, training, and evaluation code released on publication.
- **Funding:** `[to complete]`.
- **Conflicts of interest:** `[to complete]`.
- **AI-usage disclosure:** AI tools were used to assist literature organization, drafting, and code scaffolding; all methods, experiments, and claims were designed and verified by the authors.
- **Author contributions (CRediT):** `[to complete]`.

---

## References
Bangi, M.S.F., Kao, K., Kwon, J.S.-I., 2022. Physics-informed neural networks for hybrid modeling of lab-scale batch fermentation. *Chemical Engineering Research and Design* 179, 415–423.
Devarakonda, V.S., Sun, W., Tang, X., Tian, Y., 2025. Recent advances in reinforcement learning for chemical process control. *Processes* 13(6), 1791.
Downs, J.J., Vogel, E.F., 1993. A plant-wide industrial process control problem. *Computers & Chemical Engineering* 17(3), 245–255.
Fortuna, L., Graziani, S., Rizzo, A., Xibilia, M.G., 2007. *Soft Sensors for Monitoring and Control of Industrial Processes*. Springer.
Gao, J., Chen, M., Xiang, L., Xu, C., 2024. A comprehensive survey on evidential deep learning and its applications. arXiv:2409.04720.
Goldrick, S., Ştefan, A., Lovett, D., Montague, G., Lennox, B., 2015. The development of an industrial-scale fed-batch fermentation simulation. *Journal of Biotechnology* 193, 70–82.
Hinder, F., Vaquet, V., Hammer, B., 2023. One or two things we know about concept drift — a survey on monitoring evolving environments. arXiv:2310.15826.
Ji, C., Sun, W., 2022. A review on data-driven process monitoring methods: characterization and mining of industrial data. *Processes* 10(2), 335.
Liu, W., Wang, X., Owens, J.D., Li, Y., 2020. Energy-based out-of-distribution detection. *NeurIPS*.
Lobo, J.L., Laña, I., Osaba, E., Del Ser, J., 2023. On the connection between concept drift and uncertainty in industrial artificial intelligence. arXiv:2303.07940.
Lomov, I., Lyubimov, M., Makarov, I., Zhukov, L.E., 2021. Fault detection in Tennessee Eastman process with temporal deep learning models. *Journal of Industrial Information Integration* 23, 100216.
Orrù, P.F., Zoccheddu, A., Sassu, L., Mattia, C., Cozza, R., Arena, S., 2020. Machine learning approach using MLP and SVM algorithms for fault prediction of a centrifugal pump in the oil and gas industry. *Sustainability* 12(11), 4776.
Qin, S.J., 2003. Statistical process monitoring: basics and beyond. *Journal of Chemometrics* 17(8–9), 480–502.
Qu, Y., Zhou, B., Waaler, A., Cameron, D., 2023. Real-time event detection with random forests and temporal convolutional networks for a more sustainable petroleum industry. arXiv:2310.08737.
Rieth, C.A., Amsel, B.D., Tran, R., Cook, M.B., 2017. Additional Tennessee Eastman process simulation data for anomaly detection evaluation. Harvard Dataverse.
Sensoy, M., Kaplan, L., Kandemir, M., 2018. Evidential deep learning to quantify classification uncertainty. *NeurIPS*.
Sifakis, N., Sarantinoudis, N., Tsinarakis, G., Politis, C., Arampatzis, G., 2023. Soft sensing of LPG processes using deep learning. *Sensors* 23(18), 7858.
Stief, A., Tan, R., Cao, Y., Ottewill, J.R., Thornhill, N.F., Baranowski, J., 2019. A heterogeneous benchmark dataset for data analytics: multiphase flow facility case study. *Journal of Process Control* 79, 41–55.
Sun, Q., Ge, Z., 2021. A survey on deep learning for data-driven soft sensors. *IEEE Transactions on Industrial Informatics* 17(9), 5853–5866.
Wang, H., Vaze, S., Han, K., 2024. Dissecting out-of-distribution detection and open-set recognition: a critical analysis of methods and benchmarks. arXiv:2408.16757.
Wang, Y., Pan, Z., Yuan, X., Yang, C., Gui, W., 2019. A novel deep learning based fault diagnosis approach for chemical process with extended deep belief network. *ISA Transactions* 96, 457–467.
Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., Sun, L., 2023. Transformers in time series: a survey. arXiv:2202.07125.
Wu, D., Zhao, J., 2021. Process topology convolutional network model for chemical process fault diagnosis. *Process Safety and Environmental Protection* 150, 93–109.
Wu, H., Zhao, J., 2018. Deep convolutional neural network model based chemical process fault diagnosis. *Computers & Chemical Engineering* 115, 185–197.
Yang, J., Zhou, K., Li, Y., Liu, Z., 2022. Generalized out-of-distribution detection: a survey. arXiv:2110.11334.
Yuan, X., Li, L., Shardt, Y.A.W., Wang, Y., Yang, C., 2020. Deep learning with spatiotemporal attention-based LSTM for industrial soft sensor model development. *IEEE Transactions on Industrial Electronics* 68(5), 4404–4414.

*Reference list will be completed to the full 77-entry corpus and DOI-verified at citation-check stage.*
