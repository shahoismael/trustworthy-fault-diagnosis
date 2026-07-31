# Trustworthy Fault Diagnosis for Petrochemical Processes: Calibrated, Open-Set Deep Learning

## Abstract

Early, trustworthy fault detection is central to preventing accidents in petrochemical plants, yet data-driven diagnosis rarely reaches the control room: models are tuned on balanced simulated data, assume every fault is already known, and return decisions without a usable measure of confidence. We address these gaps with a single one-dimensional convolutional network that serves, unchanged, as the backbone for multi-class diagnosis, incipient-fault detection, unknown-fault rejection, and root-cause attribution. Class imbalance is handled with a focal objective, unknown faults are rejected with a feature-space Mahalanobis score, and the detection threshold is fixed on validation data so the plant operates at a stated false-alarm budget. On the canonical Tennessee Eastman benchmark the model reaches a macro-F1 of 0.88 ± 0.06 over five seeds; calibration yields a 0.90 detection rate at a 0.06 false-alarm rate with a 48-minute mean delay. Feature-space scoring separates three held-out unknown faults at 0.82 AUROC, where softmax scores stay near chance. The same network detects real seeded faults on the PRONTO pilot plant at 0.99 AUROC. We also report an honest negative result: simulation-to-plant transfer without a shared sensor space fails. Code and preprocessing pipelines are released for reproducibility.

## Keywords
Fault diagnosis; Process safety; Open-set recognition; Deep learning; Uncertainty calibration; Tennessee Eastman
