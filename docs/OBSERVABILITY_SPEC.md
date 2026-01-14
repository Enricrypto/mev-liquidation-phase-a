# Phase A — Observability & Dashboard (Draft v6)

## Metrics

- Opportunity timestamp, position ID, collateral ratio
- EV & variance, gas & slippage
- Capture probability + CI
- Failure classification & reproducibility
- Market conditions
- Cross-protocol influence

## Logging

- Structured logs via loguru
- Immutable hash-chained logs, versioned, offsite backup
- Experiment ID, run number, random seeds

## Dashboards

- Jupyter Notebook:
  - Rolling-window backtesting plots
  - Tail-event anomaly alerts
  - Minimum actionable EV thresholds
- Conservative: **informational only**, not predictive
- Optional: Streamlit for UX if needed

## Reproducibility

- Deterministic replay ≥10 seeds
- Independent verification + optional human cross-check
- Full traceability of all experiments
