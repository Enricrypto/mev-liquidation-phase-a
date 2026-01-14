# Phase A — System Architecture (Draft v6)

## Overview

- Research system to validate liquidation MEV opportunities.
- Focus on reproducibility, safety, observability.

## System Diagram

- Scripts: backtest, live simulation, replay
- Logs: hash-chained, versioned
- Dashboards: Jupyter notebook
- Data flow: historical/forked blocks → simulation → logs → analysis → promotion decision

## Core Components

- Backtest engine
- Live simulation engine
- Replay & verification
- Dashboard / reporting
- Logging & SAFE_MODE

## Data Flow

1. Fetch historical/forked blocks
2. Detect liquidatable positions
3. Simulate EV, capture probability
4. Record structured, hash-chained logs
5. Replay logs for independent verification
6. Aggregate metrics into dashboard

## State Management

- Minimal persistent state: experiment ID, random seeds
- Logs are the single source of truth
- SAFE_MODE ensures consistent runtime environment

## External Integrations

- Arbitrum RPC (forked/testnet)
- Optional APIs for gas, oracle prices
- No real capital exposure

## Security

- SAFE_MODE: runtime verification
- Private key for testnet only
- Environment variables via .env
- Logs versioned and backed up

## Observability

- Metrics: EV, gas, slippage, probability, tail-event anomalies
- Structured logs with hash chaining
- Jupyter dashboard for visualization

## Scalability & Limits

- Phase A: small number of stochastic bots (≥10)
- Tail-event stress scenarios included
- Full mainnet complexity not simulated (documented limitation)

## Out of Scope

- Real-money trading
- Production-grade execution
- Full stochastic mainnet replication
- Private relay/mempool modeling beyond documentation
