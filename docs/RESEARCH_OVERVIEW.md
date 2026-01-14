# Phase A — Research Overview (Draft v6)

## Purpose

- Empirically investigate liquidation MEV opportunities on Aave v3 (Arbitrum).
- Phase A is **research and validation only**; results **do not guarantee profitability**.
- Focus on **risk-aware, reproducible, and conservative analysis**.

## Scope

- Detect liquidatable positions with historical/forked data.
- Estimate EV under:
  - ≥10 stochastic + adversarial bots
  - Tail-event stress scenarios
  - Explicit cross-protocol and multi-chain interactions
- Use rolling-window backtesting with defined parameters.
- Collect reproducible logs and metrics.

## Non-goals

- Executing real trades with capital
- Fully replicating mainnet stochastic complexity
- Production-grade Phase B execution

## Safety

- SAFE_MODE enforced
- Testnet/forked execution only
- Minimal private key exposure
- Hash-chained, versioned logs
