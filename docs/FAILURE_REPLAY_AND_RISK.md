# Phase A — Failure Replay and Risk (Draft v6)

## Failure Taxonomy

1. Missed opportunity (stochastic/adversarial)
2. Calculation / rounding errors
3. Network latency / delays
4. Tail-event liquidity/congestion shocks
5. Cross-protocol interactions
6. Simulation stochastic variability

## Reproducibility

- Deterministic replay ≥10 seeds
- Independent verification
- Optional human cross-check
- Hash-chained, versioned, offsite logs

## Risk Controls

- Zero capital exposure
- SAFE_MODE enforced
- Testnet/forked execution only
- Maximum EV capped
- Automatic abort on anomaly, unsafe RPC, or checksum mismatch
