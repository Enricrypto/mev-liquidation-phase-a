"""Monte Carlo simulation engine for MEV analysis.

Implements the core simulation loop:
1. For each opportunity, run N iterations
2. In each iteration, all bots compete to capture the opportunity
3. Determine winner based on execution timing
4. Track EV, capture probability, and failure modes

Supports deterministic replay via random seeds.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np

from mev_analysis.core.logging import ExperimentLogger
from mev_analysis.data.models import (
    LiquidationOpportunity,
    MarketConditions,
    SimulationResult,
)
from mev_analysis.simulation.bots import Bot, BotAction, create_default_bot_pool


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    num_iterations: int = 1000
    base_seed: int = 42

    # Our simulated liquidator parameters
    our_execution_latency_ms: float = 75.0
    our_execution_latency_std: float = 25.0
    our_max_position_usd: float = 50_000.0

    # Gas estimation (in gwei)
    base_gas_used: int = 350_000
    gas_variance: int = 50_000

    # Slippage model parameters
    base_slippage_pct: float = 0.1
    slippage_per_liquidity_ratio: float = 5.0

    # Bootstrap CI parameters
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class IterationResult:
    """Result of a single simulation iteration."""

    iteration_id: int
    seed: int

    # Outcome
    our_capture: bool
    winner_bot_id: str | None
    winner_execution_time_ms: float

    # Our metrics (if we won)
    our_execution_time_ms: float
    our_ev_eth: Decimal
    our_gas_cost_eth: Decimal
    our_slippage_eth: Decimal
    our_net_profit_eth: Decimal

    # Competition metrics
    num_competing_actions: int
    bot_actions: list[BotAction] = field(default_factory=list)

    # Failure tracking
    failure_reason: str | None = None


class SimulationEngine:
    """Monte Carlo simulation engine for liquidation opportunities.

    Runs multiple iterations of competitive liquidation scenarios,
    tracking capture probability, EV distribution, and failure modes.

    Usage:
        engine = SimulationEngine(bots, config, logger)
        result = engine.simulate(opportunity)
        print(f"Capture probability: {result.capture_probability}")
    """

    def __init__(
        self,
        bots: list[Bot] | None = None,
        config: SimulationConfig | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        """Initialize simulation engine.

        Args:
            bots: List of competing bots (default: create_default_bot_pool).
            config: Simulation configuration.
            logger: Optional experiment logger.
        """
        self.config = config or SimulationConfig()
        self.bots = bots if bots is not None else create_default_bot_pool(self.config.base_seed)
        self.logger = logger
        self._rng = np.random.default_rng(self.config.base_seed)

    def simulate(
        self,
        opportunity: LiquidationOpportunity,
        market_conditions: MarketConditions | None = None,
    ) -> SimulationResult:
        """Run Monte Carlo simulation for a liquidation opportunity.

        Args:
            opportunity: The opportunity to simulate.
            market_conditions: Market conditions (uses opportunity's if None).

        Returns:
            SimulationResult with EV, capture probability, and statistics.
        """
        start_time = time.time()
        simulation_id = str(uuid.uuid4())[:8]

        market = market_conditions or opportunity.market_conditions
        iterations: list[IterationResult] = []

        if self.logger:
            self.logger.log_event(
                "simulation_started",
                {
                    "simulation_id": simulation_id,
                    "opportunity_id": opportunity.opportunity_id,
                    "num_iterations": self.config.num_iterations,
                    "num_bots": len(self.bots),
                },
            )

        # Run iterations
        for i in range(self.config.num_iterations):
            iteration_seed = self.config.base_seed + i
            result = self._run_iteration(
                iteration_id=i,
                seed=iteration_seed,
                opportunity=opportunity,
                market=market,
            )
            iterations.append(result)

        # Aggregate results
        simulation_result = self._aggregate_results(
            simulation_id=simulation_id,
            opportunity_id=opportunity.opportunity_id,
            iterations=iterations,
            duration_ms=int((time.time() - start_time) * 1000),
        )

        if self.logger:
            self.logger.log_event(
                "simulation_completed",
                {
                    "simulation_id": simulation_id,
                    "opportunity_id": opportunity.opportunity_id,
                    "capture_probability": str(simulation_result.capture_probability),
                    "mean_ev_eth": str(simulation_result.mean_ev_eth),
                    "duration_ms": simulation_result.simulation_duration_ms,
                },
            )

        return simulation_result

    def _run_iteration(
        self,
        iteration_id: int,
        seed: int,
        opportunity: LiquidationOpportunity,
        market: MarketConditions,
    ) -> IterationResult:
        """Run a single iteration of the simulation."""
        # Reset RNGs for determinism
        self._rng = np.random.default_rng(seed)
        for bot in self.bots:
            bot.reset_rng(seed + hash(bot.bot_id) % 10000)

        # Base execution time (when opportunity becomes available)
        base_time_ms = 0.0

        # Calculate our execution time
        our_latency = self._rng.normal(
            self.config.our_execution_latency_ms,
            self.config.our_execution_latency_std,
        )
        our_execution_time = base_time_ms + max(0, our_latency)

        # Get bot actions
        bot_actions: list[BotAction] = []
        for bot in self.bots:
            action = bot.execute(
                opportunity_id=opportunity.opportunity_id,
                opportunity_ev_usd=opportunity.estimated_net_profit_usd,
                max_debt_usd=opportunity.debt_to_cover_usd,
                gas_price_gwei=market.gas_price_gwei,
                current_liquidity_usd=opportunity.debt_to_cover_usd * Decimal("10"),  # Assume 10x liquidity
                base_time_ms=base_time_ms,
                liquidation_bonus_pct=Decimal("0.05"),  # 5% bonus
            )
            if action.action_type == "execute":
                bot_actions.append(action)

        # Determine winner (earliest execution)
        our_capture = True
        winner_bot_id: str | None = None
        winner_time = our_execution_time

        for action in bot_actions:
            if action.timestamp_ms < winner_time:
                winner_time = action.timestamp_ms
                winner_bot_id = action.bot_id
                our_capture = False

        # Calculate our metrics
        our_ev_eth, our_gas_cost_eth, our_slippage_eth, our_net_profit_eth = (
            self._calculate_our_metrics(opportunity, market, our_capture)
        )

        # Determine failure reason if we didn't capture
        failure_reason: str | None = None
        if not our_capture:
            winner_action = next(
                (a for a in bot_actions if a.bot_id == winner_bot_id), None
            )
            if winner_action:
                failure_reason = f"outcompeted_by_{winner_action.bot_type.value}"

        return IterationResult(
            iteration_id=iteration_id,
            seed=seed,
            our_capture=our_capture,
            winner_bot_id=winner_bot_id,
            winner_execution_time_ms=winner_time,
            our_execution_time_ms=our_execution_time,
            our_ev_eth=our_ev_eth,
            our_gas_cost_eth=our_gas_cost_eth,
            our_slippage_eth=our_slippage_eth,
            our_net_profit_eth=our_net_profit_eth,
            num_competing_actions=len(bot_actions),
            bot_actions=bot_actions,
            failure_reason=failure_reason,
        )

    def _calculate_our_metrics(
        self,
        opportunity: LiquidationOpportunity,
        market: MarketConditions,
        captured: bool,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """Calculate our EV, costs, and profit for an iteration."""
        if not captured:
            return Decimal(0), Decimal(0), Decimal(0), Decimal(0)

        # EV from opportunity (gross profit)
        ev_usd = opportunity.estimated_gross_profit_usd

        # Gas cost
        gas_used = self.config.base_gas_used + int(
            self._rng.normal(0, self.config.gas_variance)
        )
        gas_used = max(200_000, gas_used)  # Minimum gas
        gas_cost_eth = Decimal(str(gas_used)) * market.gas_price_gwei / Decimal("1e9")

        # Slippage
        position_ratio = float(opportunity.debt_to_cover_usd) / 100_000  # vs 100k liquidity
        slippage_pct = self.config.base_slippage_pct + position_ratio * self.config.slippage_per_liquidity_ratio
        slippage_pct += self._rng.normal(0, 0.1)
        slippage_pct = max(0, slippage_pct)

        slippage_usd = opportunity.collateral_to_receive_usd * Decimal(str(slippage_pct / 100))

        # Convert to ETH
        eth_price = market.eth_price_usd if market.eth_price_usd > 0 else Decimal("2000")
        ev_eth = ev_usd / eth_price
        slippage_eth = slippage_usd / eth_price

        # Net profit
        net_profit_eth = ev_eth - gas_cost_eth - slippage_eth

        return ev_eth, gas_cost_eth, slippage_eth, net_profit_eth

    def _aggregate_results(
        self,
        simulation_id: str,
        opportunity_id: str,
        iterations: list[IterationResult],
        duration_ms: int,
    ) -> SimulationResult:
        """Aggregate iteration results into simulation summary."""
        # Extract metrics
        captures = [it.our_capture for it in iterations]
        profits = [float(it.our_net_profit_eth) for it in iterations]

        # Capture probability
        capture_count = sum(captures)
        capture_prob = Decimal(str(capture_count / len(iterations)))

        # EV statistics (only for captured iterations, or all with 0 for non-capture)
        profits_array = np.array(profits)
        mean_ev = Decimal(str(np.mean(profits_array)))
        std_ev = Decimal(str(np.std(profits_array)))
        min_ev = Decimal(str(np.min(profits_array)))
        max_ev = Decimal(str(np.max(profits_array)))

        # Bootstrap confidence intervals
        ev_ci_lower, ev_ci_upper = self._bootstrap_ci(profits_array, self.config.bootstrap_samples)
        capture_ci_lower, capture_ci_upper = self._bootstrap_ci(
            np.array([1 if c else 0 for c in captures]),
            self.config.bootstrap_samples,
        )

        # Cost breakdown (average across captured iterations)
        captured_iterations = [it for it in iterations if it.our_capture]
        if captured_iterations:
            mean_gas = Decimal(str(np.mean([float(it.our_gas_cost_eth) for it in captured_iterations])))
            mean_slippage = Decimal(str(np.mean([float(it.our_slippage_eth) for it in captured_iterations])))
        else:
            mean_gas = Decimal(0)
            mean_slippage = Decimal(0)

        # Failure analysis
        failure_reasons: dict[str, int] = {}
        for it in iterations:
            if it.failure_reason:
                failure_reasons[it.failure_reason] = failure_reasons.get(it.failure_reason, 0) + 1

        failure_count = len(iterations) - capture_count
        failure_rate = Decimal(str(failure_count / len(iterations)))

        return SimulationResult(
            opportunity_id=opportunity_id,
            simulation_id=simulation_id,
            num_iterations=len(iterations),
            random_seed=self.config.base_seed,
            num_competing_bots=len(self.bots),
            mean_ev_eth=mean_ev,
            std_ev_eth=std_ev,
            min_ev_eth=min_ev,
            max_ev_eth=max_ev,
            ev_ci_lower_95=Decimal(str(ev_ci_lower)),
            ev_ci_upper_95=Decimal(str(ev_ci_upper)),
            capture_probability=capture_prob,
            capture_ci_lower_95=Decimal(str(max(0, capture_ci_lower))),
            capture_ci_upper_95=Decimal(str(min(1, capture_ci_upper))),
            mean_gas_cost_eth=mean_gas,
            mean_slippage_eth=mean_slippage,
            failure_count=failure_count,
            failure_rate=failure_rate,
            failure_reasons=failure_reasons,
            simulation_duration_ms=duration_ms,
        )

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        n_samples: int,
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(data) == 0:
            return 0.0, 0.0

        bootstrap_means = []
        for _ in range(n_samples):
            sample = self._rng.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return float(lower), float(upper)


def run_simulation_batch(
    opportunities: list[LiquidationOpportunity],
    engine: SimulationEngine,
    logger: ExperimentLogger | None = None,
) -> list[SimulationResult]:
    """Run simulation for multiple opportunities.

    Args:
        opportunities: List of opportunities to simulate.
        engine: Configured simulation engine.
        logger: Optional logger.

    Returns:
        List of simulation results.
    """
    results: list[SimulationResult] = []

    for i, opp in enumerate(opportunities):
        if logger:
            logger.info(
                f"Simulating opportunity {i + 1}/{len(opportunities)}",
                {"opportunity_id": opp.opportunity_id},
            )

        result = engine.simulate(opp)
        results.append(result)

        if logger:
            logger.log_metric(
                "batch_progress",
                i + 1,
                {
                    "total": len(opportunities),
                    "opportunity_id": opp.opportunity_id,
                    "capture_probability": str(result.capture_probability),
                },
            )

    return results
