# Phase A — Experiment Design (Draft v6)

## Backtesting

- Historical/forked blocks only
- Randomized, stratified controls
- Stochastic + adversarial bots + tail-event stress
- EV, gas, slippage, collateral ratio recorded

## Live Simulation

- Testnet/forked mainnet only
- Dry-run, Monte Carlo + tail-event stress
- Explicit cross-protocol interactions

## Replay & Reproducibility

- Deterministic replay ≥10 seeds
- Independent verification
- Optional human cross-check
- Hash-chained, versioned logs

## Metrics Collected

- Opportunity timestamp, position ID, collateral ratio
- EV & variance, gas & slippage
- Capture probability + CI
- Failure type & reproducibility
- Market conditions
- Cross-protocol influence

## Statistical Rigor

- Bootstrap CIs
- Minimum sample size enforced
- Multiple hypothesis correction
- Tail-event stress testing
- Rolling-window validation

## Safety Controls

- SAFE_MODE enforced
- Automatic abort on unsafe RPC/env/private key
- Maximum EV per simulation capped
