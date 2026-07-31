# PICUP-FDD — Research Title (FINAL, evidence-matched)

## Final title
**Trustworthy Deep Fault Diagnosis for Petrochemical Processes: Calibrated, Open-Set Recognition of Known and Unknown Faults**

## Shorter alternative
**Trustworthy Deep Fault Diagnosis for Petrochemical Processes: Open-Set Recognition with Controlled False Alarms**

## Why this title (every word is backed by results)
- "Deep Fault Diagnosis / Petrochemical" — unified CNN, TEP macro-F1 0.88 ± 0.03.
- "Trustworthy / Calibrated / Controlled False Alarms" — validation-calibrated operating point: FDR 0.90 at FAR 0.06, detection delay ≈ 48 min.
- "Open-Set / Known and Unknown Faults" — Mahalanobis open-set AUROC 0.82 (0.845 ± 0.006).
- Interpretability (xmeas_21 cooling temp) and incipient-fault recovery support the "trustworthy" framing.

## Removed from the title (not supported by the simulation — honesty)
- ~~Physics-Informed~~ — no physics-informed component was implemented; would be a false claim.
- ~~Simulation-to-Real Transfer~~ — TEP→PRONTO transfer fails (AUROC 0.44); kept in the paper only as an honest negative result / limitation, NOT a headline claim.
- ~~Uncertainty-Aware (as headline)~~ — evidential model underperformed; uncertainty is delivered via entropy/Mahalanobis, so it is a supporting result, not the lead.

_Status: FINAL. Target: Elsevier/Springer Q1._
