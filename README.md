# PICUP-FDD

**Physics-Informed, Calibrated, Uncertainty-aware, Open-set framework for Fault Diagnosis and soft sensing in petrochemical processes.**

A research project at the intersection of machine learning (classification + optimization) and chemical/petrochemical process engineering, built entirely on open-access datasets.

## Core idea
Train fault-diagnosis models on the simulated Tennessee Eastman Process (TEP) benchmark and validate on the *real* Cranfield multiphase-flow facility, with:
- class-imbalance robustness (rare/incipient faults)
- calibrated uncertainty + open-set rejection of unknown faults
- simulation→real domain adaptation
- interpretable, process-topology-aware root-cause attribution

See `docs/AI_ChemEng_Research_Gap_and_Proposal.md` for the full gap analysis and proposal.

## Structure
```
PICUP-FDD/
├── docs/          Proposal, gap analysis, notes
├── data/          Raw + processed datasets (TEP, Cranfield, SRU)
├── notebooks/     Exploration
├── src/           Models, training, evaluation
├── results/       Figures, metrics, tables
└── references/    Papers, bibliography
```

## Datasets (open access)
- **Extended TEP** — Rieth et al. 2017, Harvard Dataverse, DOI `10.7910/DVN/6C3JR1`
- **Cranfield Three-Phase Flow Facility** — Ruiz-Cárcel et al. 2015, IEEE DataPort
- **Debutanizer + SRU soft-sensor benchmark** — Fortuna et al. 2007

## Status
Proposal drafted (July 2026). Next: acquire datasets → reproduce baselines → add contributions.
