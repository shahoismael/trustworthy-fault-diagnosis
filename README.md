# trustworthy-fault-diagnosis

**Trustworthy Deep Fault Diagnosis for Chemical Process Monitoring: Calibrated Detection and Open-Set Recognition of Known and Unknown Faults.**

One 1-D convolutional backbone, trained once, provides every component of a deployable fault-detection and diagnosis (FDD) system: multi-class diagnosis, incipient-fault recovery, calibrated detection at a stated false-alarm budget, open-set rejection of unknown faults, and input-gradient interpretability. Built entirely on open-access datasets.

## What the framework does
- **Diagnosis** with a focal-loss CNN, evaluated under the canonical Tennessee Eastman train/test protocol (train on the training simulations, test on the separate testing simulations).
- **Calibrated detection** — the decision threshold is fixed on validation normal windows to a stated false-alarm budget, and fault-detection rate, false-alarm rate, and detection delay are reported at that operating point.
- **Open-set rejection** — unknown faults are scored in the network's feature space (Mahalanobis and k-NN), benchmarked head-to-head against softmax, energy, conformal, and classical PCA T²/SPE scores computed from the same model.
- **Interpretability** — input-gradient attribution over the same network.
- **Honest limits** — simulation-to-real weight transfer is reported as a negative result; auxiliary datasets use task-appropriate models.

## Headline results
| Result | Value |
|---|---|
| TEP diagnosis, macro-F1 (5 seeds, canonical test) | 0.88 ± 0.06 |
| Calibrated detection | FDR 0.90 at FAR 0.06, mean delay 47.8 min |
| Open-set AUROC — Mahalanobis / k-NN (canonical triple) | 0.83 / 0.89 |
| Open-set AUROC — Mahalanobis / k-NN (5 rotated held-out sets) | 0.88 ± 0.07 / 0.91 ± 0.05 |
| Open-set AUROC — softmax family | ≈ chance, unstable (0.47–0.56) |
| Real PRONTO plant, AUROC (5 seeds) | 0.985 ± 0.008 |
| Simulation→real transfer (negative result) | AUROC 0.34–0.44 |

## Datasets (all open access)
- **Extended Tennessee Eastman** — Rieth et al. 2017, Harvard Dataverse, DOI `10.7910/DVN/6C3JR1` (primary).
- **PRONTO multiphase-flow facility** — Stief et al. 2019, Zenodo (real-plant validation).
- **IndPenSim** fed-batch penicillin — Goldrick et al. 2015, Mendeley Data (auxiliary).
- **Debutanizer + Sulphur Recovery Unit** soft-sensor benchmark — Fortuna et al. 2007 (auxiliary).
- **Steel Plates Faults** — UCI Machine Learning Repository (cross-domain probe).

## Repository structure
```
├── src/          Numbered pipeline: step01–step23 (preprocess, baselines, unified CNN,
│                 ablation, multi-seed, open-set + rotation, PCA T²/SPE, PRONTO, soft sensor, …)
├── data/         Raw + processed datasets
├── results/      Metrics CSVs and figure arrays
├── docs/         Proposal and gap analysis
└── references/   Bibliography
```

## Reproduce
```bash
pip install -r requirements.txt
python src/step02_preprocess_tep.py          # build the canonical TEP cache
python src/step18_multiseed_tep.py           # macro-F1 over 5 seeds
python src/step17_ablation.py                # focal-loss / class-weight ablation
python src/step09_unified_cnn.py             # unified model: detection + open-set + saliency
python src/step19_openset_baselines.py       # MSP/energy/entropy/KNN/conformal/Mahalanobis
python src/step20_baselines_canonical_known.py  # baselines on canonical test, known classes
python src/step21_pca_openset.py             # classical PCA T²/SPE open-set comparator
python src/step22_openset_rotation.py        # open-set AUROC over rotated held-out fault sets
python src/step15_cnn_pronto.py              # same CNN on the real PRONTO plant
python src/step23_cnn_pronto_multiseed.py    # PRONTO over 5 seeds (dispersion)
```
Every number in the paper's tables is produced by these scripts; outputs land in `results/`.

## Citation
Hassen, S. I., & Abdlrazaq, A. A. (2026). *Trustworthy Deep Fault Diagnosis for Chemical Process Monitoring: Calibrated Detection and Open-Set Recognition of Known and Unknown Faults.* (Under review.)

## License
Code released for reproducibility. See `LICENSE` (add one before publication, e.g. MIT).
