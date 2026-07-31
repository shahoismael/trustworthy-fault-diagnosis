# data/ — dataset plan (locked for Q1 submission)

**Best single (primary) dataset:** Extended Tennessee Eastman Process (TEP) — the field-standard chemical-process fault benchmark. Everything else supports it.

## Optimal set (4 datasets — meets the "3–5 for Q1" bar with real sim→real diversity)

| Role | Dataset | Why | Access |
|------|---------|-----|--------|
| **Primary — simulated FDD** | **Extended TEP** (Rieth et al., 2017) — 20 faults, 500 runs, 52 vars | Core benchmark; supports imbalance, rare/incipient faults, open-set | Harvard Dataverse, DOI 10.7910/DVN/6C3JR1 · CSV mirror: Kaggle `afrniomelo/tep-csv` |
| **Real plant — sim→real** | **Cranfield Three-Phase Flow Facility** (Ruiz-Cárcel et al., 2015) — 6 seeded faults, real noise | Delivers the sim→real novelty; real petrochemical/upstream data | IEEE DataPort "Three-Phase Flow Facility" |
| **Cross-simulator transfer** | **Original TEP** (Downs–Vogel; Braatz MATLAB simulator) | Same process, different simulator/control strategy → tests domain shift without new data | Free (Braatz group / GitHub) |
| **Process diversity** | **IndPenSim** fed-batch penicillin (Goldrick et al.) | A batch bioprocess beyond TEP; strengthens generality claim reviewers ask for | Open (IEEE DataPort / industrialpenicillinsimulation.com) |

**Optional 5th (only if keeping the soft-sensor track 6B):** Fortuna Debutanizer + SRU soft-sensor benchmark (real refinery, regression).

## Rationale
- 2 simulated chemical (TEP variants) + 1 real multiphase (Cranfield) + 1 batch bioprocess (IndPenSim) = strong diversity across process type, simulated vs real, and operating regime — exactly what Q1 reviewers look for.
- Primary results and ablations run on Extended TEP; Cranfield carries the sim→real claim; Original TEP and IndPenSim demonstrate generalization.

## Folder layout
```
data/
├── tep_extended/     raw/ + processed/
├── cranfield/        raw/ + processed/
├── tep_original/     raw/ + processed/
├── indpensim/        raw/ + processed/
└── softsensor/       (optional: debutanizer + SRU)
```
Keep large raw files out of version control.
