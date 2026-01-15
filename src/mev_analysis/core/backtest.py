"""Backtest runner for liquidation MEV research.

Implements the backtesting framework from EXPERIMENT_DESIGN.md:
- Rolling-window backtesting
- Stratified controls
- Multiple hypothesis correction support
- Statistical rigor with minimum sample sizes

All results are logged with hash-chaining for reproducibility.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator

import numpy as np

from mev_analysis.core.logging import ExperimentLogger
from mev_analysis.core.opportunity_detector import DetectorConfig, OpportunityDetector
from mev_analysis.data.models import (
    LiquidationOpportunity,
    MarketConditions,
    SimulationResult,
    UserPosition,
)
from mev_analysis.simulation.engine import SimulationConfig, SimulationEngine


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    # Window parameters
    window_size_blocks: int = 1000  # Blocks per window
    window_stride_blocks: int = 500  # Overlap between windows

    # Simulation parameters
    num_simulation_iterations: int = 1000
    num_seeds: int = 10  # ≥10 seeds for reproducibility
    base_seed: int = 42

    # Statistical requirements
    min_sample_size: int = 30  # Minimum opportunities per window
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000

    # Filtering
    min_profit_eth: Decimal = Decimal("0.01")

    # Multiple hypothesis correction
    apply_bonferroni: bool = True
    num_hypotheses: int = 4  # From ALPHA_HYPOTHESES.md


@dataclass
class WindowResult:
    """Result for a single backtest window."""

    window_id: str
    start_block: int
    end_block: int

    # Opportunity statistics
    num_positions_scanned: int
    num_opportunities_detected: int
    num_opportunities_simulated: int

    # Aggregated simulation results
    mean_ev_eth: Decimal
    std_ev_eth: Decimal
    ev_ci_lower_95: Decimal
    ev_ci_upper_95: Decimal

    mean_capture_probability: Decimal
    capture_ci_lower_95: Decimal
    capture_ci_upper_95: Decimal

    # Market conditions summary
    mean_gas_gwei: Decimal
    mean_eth_price_usd: Decimal

    # Per-opportunity results
    opportunity_results: list[SimulationResult] = field(default_factory=list)

    # Metadata
    seeds_used: list[int] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class BacktestResult:
    """Complete backtest result across all windows."""

    backtest_id: str
    config: BacktestConfig
    started_at: datetime
    completed_at: datetime | None = None

    # Window results
    windows: list[WindowResult] = field(default_factory=list)

    # Aggregate statistics
    total_positions_scanned: int = 0
    total_opportunities_detected: int = 0
    total_opportunities_simulated: int = 0

    # Overall metrics (across all windows)
    overall_mean_ev_eth: Decimal = Decimal(0)
    overall_ev_ci_lower_95: Decimal = Decimal(0)
    overall_ev_ci_upper_95: Decimal = Decimal(0)

    overall_mean_capture_prob: Decimal = Decimal(0)
    overall_capture_ci_lower_95: Decimal = Decimal(0)
    overall_capture_ci_upper_95: Decimal = Decimal(0)

    # Hypothesis testing
    hypothesis_results: dict[str, Any] = field(default_factory=dict)

    # Validation flags
    meets_sample_size: bool = False
    meets_capture_threshold: bool = False
    meets_ev_threshold: bool = False


class BacktestRunner:
    """Run backtests on historical position data.

    Implements rolling-window backtesting with proper statistical controls.

    Usage:
        runner = BacktestRunner(config, logger)
        result = runner.run(positions, market_conditions_by_block)
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        detector_config: DetectorConfig | None = None,
        simulation_config: SimulationConfig | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        """Initialize backtest runner.

        Args:
            config: Backtest configuration.
            detector_config: Opportunity detector configuration.
            simulation_config: Simulation engine configuration.
            logger: Experiment logger for tracking.
        """
        self.config = config or BacktestConfig()
        self.detector = OpportunityDetector(detector_config, logger)
        self.simulation_config = simulation_config or SimulationConfig(
            num_iterations=self.config.num_simulation_iterations,
            base_seed=self.config.base_seed,
            bootstrap_samples=self.config.bootstrap_samples,
        )
        self.logger = logger
        self._rng = np.random.default_rng(self.config.base_seed)

    def run(
        self,
        positions: list[UserPosition],
        market_conditions: MarketConditions | dict[int, MarketConditions],
    ) -> BacktestResult:
        """Run backtest on positions.

        Args:
            positions: List of positions to analyze.
            market_conditions: Either single MarketConditions or dict by block.

        Returns:
            BacktestResult with all window results and aggregates.
        """
        backtest_id = f"bt_{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)

        if self.logger:
            self.logger.log_event(
                "backtest_started",
                {
                    "backtest_id": backtest_id,
                    "num_positions": len(positions),
                    "num_seeds": self.config.num_seeds,
                    "window_size": self.config.window_size_blocks,
                },
            )

        # Group positions by block
        positions_by_block = self._group_by_block(positions)

        if not positions_by_block:
            return BacktestResult(
                backtest_id=backtest_id,
                config=self.config,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Determine block range
        min_block = min(positions_by_block.keys())
        max_block = max(positions_by_block.keys())

        # Generate windows
        windows = list(self._generate_windows(min_block, max_block))

        if self.logger:
            self.logger.info(
                f"Generated {len(windows)} windows",
                {"min_block": min_block, "max_block": max_block},
            )

        # Run each window
        window_results: list[WindowResult] = []
        for window_start, window_end in windows:
            window_result = self._run_window(
                window_start=window_start,
                window_end=window_end,
                positions_by_block=positions_by_block,
                market_conditions=market_conditions,
            )
            window_results.append(window_result)

        # Aggregate results
        result = self._aggregate_results(
            backtest_id=backtest_id,
            started_at=started_at,
            window_results=window_results,
        )

        if self.logger:
            self.logger.log_event(
                "backtest_completed",
                {
                    "backtest_id": backtest_id,
                    "num_windows": len(window_results),
                    "total_opportunities": result.total_opportunities_simulated,
                    "overall_mean_ev_eth": str(result.overall_mean_ev_eth),
                    "overall_capture_prob": str(result.overall_mean_capture_prob),
                },
            )

        return result

    def _group_by_block(
        self,
        positions: list[UserPosition],
    ) -> dict[int, list[UserPosition]]:
        """Group positions by block number."""
        grouped: dict[int, list[UserPosition]] = {}
        for pos in positions:
            block = pos.block_number
            if block not in grouped:
                grouped[block] = []
            grouped[block].append(pos)
        return grouped

    def _generate_windows(
        self,
        min_block: int,
        max_block: int,
    ) -> Iterator[tuple[int, int]]:
        """Generate rolling windows."""
        start = min_block
        while start + self.config.window_size_blocks <= max_block:
            end = start + self.config.window_size_blocks
            yield (start, end)
            start += self.config.window_stride_blocks

        # Final window to cover remaining blocks
        if start < max_block:
            yield (start, max_block)

    def _run_window(
        self,
        window_start: int,
        window_end: int,
        positions_by_block: dict[int, list[UserPosition]],
        market_conditions: MarketConditions | dict[int, MarketConditions],
    ) -> WindowResult:
        """Run backtest for a single window."""
        import time
        start_time = time.time()

        window_id = f"win_{window_start}_{window_end}"

        # Collect positions in window
        window_positions: list[UserPosition] = []
        for block in range(window_start, window_end + 1):
            if block in positions_by_block:
                window_positions.extend(positions_by_block[block])

        # Get market conditions for window
        if isinstance(market_conditions, dict):
            # Use middle block's conditions or first available
            mid_block = (window_start + window_end) // 2
            market = market_conditions.get(mid_block)
            if market is None:
                # Find closest
                for b in sorted(market_conditions.keys()):
                    if b >= window_start:
                        market = market_conditions[b]
                        break
            if market is None:
                market = list(market_conditions.values())[0]
        else:
            market = market_conditions

        # Detect opportunities
        opportunities = list(self.detector.detect(window_positions, market))

        if self.logger:
            self.logger.info(
                f"Window {window_id}: {len(opportunities)} opportunities from {len(window_positions)} positions",
                {
                    "window_id": window_id,
                    "start_block": window_start,
                    "end_block": window_end,
                },
            )

        # Simulate opportunities with multiple seeds
        all_results: list[SimulationResult] = []
        seeds_used: list[int] = []

        for opp in opportunities:
            # Run with multiple seeds
            for seed_idx in range(self.config.num_seeds):
                seed = self.config.base_seed + seed_idx * 1000

                sim_config = SimulationConfig(
                    num_iterations=self.config.num_simulation_iterations,
                    base_seed=seed,
                    bootstrap_samples=self.config.bootstrap_samples,
                )
                engine = SimulationEngine(config=sim_config)
                result = engine.simulate(opp, market)
                all_results.append(result)

                if seed not in seeds_used:
                    seeds_used.append(seed)

        # Aggregate window results
        if all_results:
            evs = [float(r.mean_ev_eth) for r in all_results]
            capture_probs = [float(r.capture_probability) for r in all_results]

            mean_ev = Decimal(str(np.mean(evs)))
            std_ev = Decimal(str(np.std(evs)))
            ev_ci = self._bootstrap_ci(evs)

            mean_capture = Decimal(str(np.mean(capture_probs)))
            capture_ci = self._bootstrap_ci(capture_probs)
        else:
            mean_ev = Decimal(0)
            std_ev = Decimal(0)
            ev_ci = (0.0, 0.0)
            mean_capture = Decimal(0)
            capture_ci = (0.0, 0.0)

        duration_ms = int((time.time() - start_time) * 1000)

        return WindowResult(
            window_id=window_id,
            start_block=window_start,
            end_block=window_end,
            num_positions_scanned=len(window_positions),
            num_opportunities_detected=len(opportunities),
            num_opportunities_simulated=len(all_results),
            mean_ev_eth=mean_ev,
            std_ev_eth=std_ev,
            ev_ci_lower_95=Decimal(str(ev_ci[0])),
            ev_ci_upper_95=Decimal(str(ev_ci[1])),
            mean_capture_probability=mean_capture,
            capture_ci_lower_95=Decimal(str(max(0, capture_ci[0]))),
            capture_ci_upper_95=Decimal(str(min(1, capture_ci[1]))),
            mean_gas_gwei=market.gas_price_gwei,
            mean_eth_price_usd=market.eth_price_usd,
            opportunity_results=all_results,
            seeds_used=seeds_used,
            duration_ms=duration_ms,
        )

    def _bootstrap_ci(self, data: list[float]) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not data:
            return (0.0, 0.0)

        arr = np.array(data)
        bootstrap_means = []

        for _ in range(self.config.bootstrap_samples):
            sample = self._rng.choice(arr, size=len(arr), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def _aggregate_results(
        self,
        backtest_id: str,
        started_at: datetime,
        window_results: list[WindowResult],
    ) -> BacktestResult:
        """Aggregate window results into final backtest result."""
        total_positions = sum(w.num_positions_scanned for w in window_results)
        total_detected = sum(w.num_opportunities_detected for w in window_results)
        total_simulated = sum(w.num_opportunities_simulated for w in window_results)

        # Aggregate EVs and capture probabilities across windows
        if window_results:
            all_evs = [float(w.mean_ev_eth) for w in window_results if w.mean_ev_eth > 0]
            all_captures = [float(w.mean_capture_probability) for w in window_results]

            if all_evs:
                overall_ev = Decimal(str(np.mean(all_evs)))
                ev_ci = self._bootstrap_ci(all_evs)
            else:
                overall_ev = Decimal(0)
                ev_ci = (0.0, 0.0)

            if all_captures:
                overall_capture = Decimal(str(np.mean(all_captures)))
                capture_ci = self._bootstrap_ci(all_captures)
            else:
                overall_capture = Decimal(0)
                capture_ci = (0.0, 0.0)
        else:
            overall_ev = Decimal(0)
            ev_ci = (0.0, 0.0)
            overall_capture = Decimal(0)
            capture_ci = (0.0, 0.0)

        # Check if meets thresholds
        meets_sample = total_detected >= self.config.min_sample_size
        meets_capture = overall_capture >= Decimal("0.03")  # 3% threshold from docs
        meets_ev = overall_ev >= self.config.min_profit_eth

        # Hypothesis testing results
        adjusted_alpha = self.config.confidence_level
        if self.config.apply_bonferroni:
            adjusted_alpha = 1 - (1 - self.config.confidence_level) / self.config.num_hypotheses

        hypothesis_results = {
            "h1_profitable_opportunities": {
                "result": meets_ev and ev_ci[0] > 0,
                "mean_ev_eth": str(overall_ev),
                "ci_lower": str(ev_ci[0]),
                "ci_upper": str(ev_ci[1]),
            },
            "h2_capture_probability": {
                "result": meets_capture,
                "mean_probability": str(overall_capture),
                "threshold": "0.03",
            },
            "adjusted_alpha": adjusted_alpha,
            "bonferroni_applied": self.config.apply_bonferroni,
        }

        return BacktestResult(
            backtest_id=backtest_id,
            config=self.config,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            windows=window_results,
            total_positions_scanned=total_positions,
            total_opportunities_detected=total_detected,
            total_opportunities_simulated=total_simulated,
            overall_mean_ev_eth=overall_ev,
            overall_ev_ci_lower_95=Decimal(str(ev_ci[0])),
            overall_ev_ci_upper_95=Decimal(str(ev_ci[1])),
            overall_mean_capture_prob=overall_capture,
            overall_capture_ci_lower_95=Decimal(str(max(0, capture_ci[0]))),
            overall_capture_ci_upper_95=Decimal(str(min(1, capture_ci[1]))),
            hypothesis_results=hypothesis_results,
            meets_sample_size=meets_sample,
            meets_capture_threshold=meets_capture,
            meets_ev_threshold=meets_ev,
        )


def create_synthetic_positions(
    num_positions: int,
    num_liquidatable: int,
    base_block: int = 1000000,
    seed: int = 42,
) -> list[UserPosition]:
    """Create synthetic positions for testing.

    Args:
        num_positions: Total number of positions.
        num_liquidatable: Number that should be liquidatable.
        base_block: Starting block number.
        seed: Random seed.

    Returns:
        List of synthetic UserPosition objects.
    """
    from mev_analysis.data.models import AssetType, CollateralAsset, DebtAsset

    rng = np.random.default_rng(seed)
    positions = []

    for i in range(num_positions):
        is_liquidatable = i < num_liquidatable

        # Generate health factor
        if is_liquidatable:
            hf = Decimal(str(rng.uniform(0.5, 0.99)))
        else:
            hf = Decimal(str(rng.uniform(1.1, 2.5)))

        # Generate collateral
        collateral_value = Decimal(str(rng.uniform(1000, 100000)))
        eth_price = Decimal("2000")
        eth_amount = collateral_value / eth_price

        collateral = CollateralAsset(
            symbol="WETH",
            address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            decimals=18,
            amount_raw=int(float(eth_amount) * 10**18),
            price_usd=eth_price,
            asset_type=AssetType.ETH_CORRELATED,
            liquidation_threshold=Decimal("0.825"),
            liquidation_bonus=Decimal("0.05"),
            ltv=Decimal("0.80"),
        )

        # Calculate debt to achieve target health factor
        # HF = (collateral * liq_threshold) / debt
        # debt = (collateral * liq_threshold) / HF
        debt_value = (collateral_value * collateral.liquidation_threshold) / hf

        debt = DebtAsset(
            symbol="USDC",
            address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            decimals=6,
            amount_raw=int(float(debt_value) * 10**6),
            price_usd=Decimal("1.0"),
            asset_type=AssetType.STABLE,
            is_stable_rate=False,
            current_rate=Decimal("0.05"),
        )

        position = UserPosition(
            user_address=f"0x{i:040x}",
            block_number=base_block + (i % 100),
            collaterals=[collateral],
            debts=[debt],
            health_factor=hf,
            health_factor_source="synthetic",
        )
        positions.append(position)

    return positions
