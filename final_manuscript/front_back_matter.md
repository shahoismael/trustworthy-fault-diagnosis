# Front & Back Matter

## Authors
Shaho Ismael Hassen ^a,^* , Ahmed Abdulfatah Abdlrazaq ^b

^a Department of Chemical and Petrochemical Engineering, College of Engineering, Salahaddin University-Erbil, Erbil, Iraq. ORCID: 0000-0002-6403-7748
^b Directorate of Information Technology, Salahaddin University-Erbil, Erbil, Iraq. ORCID: 0000-0002-3054-045X

*Corresponding author: Shaho Ismael Hassen — shaho.hassen@su.edu.krd
Co-author email: ahmed.abdulfatah@su.edu.krd


## Highlights (for Elsevier submission; 3–5 bullets, ≤85 chars each)
- One CNN backbone handles diagnosis, open-set rejection, and interpretation.
- Validation-calibrated threshold gives FDR 0.90 at FAR 0.06 on Tennessee Eastman.
- Feature-space Mahalanobis rejects unknown faults (AUROC 0.82); softmax fails.
- Same network detects real PRONTO plant faults at AUROC 0.99.
- Sim-to-real transfer reported as an honest negative result.

## Graphical abstract (description)
The unified backbone (centre) feeding four outputs — diagnosis, calibrated detection, open-set rejection, attribution — with a Tennessee Eastman schematic on one side and the PRONTO real-plant photo on the other, connected by "same architecture."

---

## Declarations

**Data availability.** All five datasets are open access: Extended Tennessee Eastman (Harvard Dataverse, DOI 10.7910/DVN/6C3JR1); PRONTO multiphase-flow facility (Zenodo); IndPenSim (Mendeley Data); Debutanizer and SRU soft-sensor benchmark (Fortuna et al., 2007); Steel Plates Faults (UCI). Exact sources are listed in Section 3.1.

**Code availability.** Preprocessing, training, and evaluation code, together with the exact configuration used to produce every table, are released at the project repository upon publication.

**Ethics statement.** This study uses only simulated and instrumented-equipment process data. It involves no human participants, animal subjects, or personal data; no ethics approval was required.

**Funding.** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Conflicts of interest.** The authors declare no competing financial or personal interests.

**Author contributions (CRediT).**
Shaho Ismael Hassen: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization, Project administration.
Ahmed Abdulfatah Abdlrazaq: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization.

**Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.** During the preparation of this work the authors used a generative AI assistant in order to organize the literature, scaffold analysis code, and improve the language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

---

## Limitations (cross-reference)
The study's limitations are stated in full in Section 5.8: (i) simulation-to-real weight transfer is infeasible without a shared sensor space and is reported as a negative result; (ii) incipient-fault recall varies across random seeds; (iii) open-set evaluation uses a single held-out unknown-fault set; (iv) uncertainty is delivered by a feature-space distance rather than an evidential head; (v) auxiliary datasets use task-appropriate models and the IndPenSim result is sample-level.

## Reproducibility note
Tennessee Eastman classification figures are computed on the canonical held-out testing simulations (train on `*_Training`, test on `*_Testing`, fault onset at sample 160). The classification head is reported over five seeds {42, 7, 123, 2024, 2025}: macro-F1 0.88 ± 0.06 (95% CI ±0.05); rare-fault recall F3 0.73 ± 0.41, F9 0.79 ± 0.16, F15 0.26 ± 0.24 (`results/tep_multiseed_classification.csv`). Open-set and auxiliary results likewise carry multi-seed intervals.
