"""Tests for simulation engine."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mev_analysis.data.models import (
    LiquidationOpportunity,
    MarketConditions,
    UserPosition,
)
from mev_analysis.simulation.bots import BotType, create_bot, create_default_bot_pool
from mev_analysis.simulation.engine import (
    SimulationConfig,
    SimulationEngine,
    run_simulation_batch,
)


@pytest.fixture
def sample_opportunity() -> LiquidationOpportunity:
    """Create a sample liquidation opportunity."""
    position = UserPosition(
        user_address="0x1234567890abcdef1234567890abcdef12345678",
        block_number=12345678,
        collaterals=[],
        debts=[],
    )

    market = MarketConditions(
        block_number=12345678,
        timestamp=datetime.now(timezone.utc),
        gas_price_gwei=Decimal("0.1"),
        eth_price_usd=Decimal("2000"),
    )

    return LiquidationOpportunity(
        opportunity_id="test_opp_1",
        detected_at_block=12345678,
        detected_at_timestamp=datetime.now(timezone.utc),
        user_address="0x1234567890abcdef1234567890abcdef12345678",
        position_snapshot=position,
        market_conditions=market,
        debt_to_cover_usd=Decimal("10000"),
        debt_asset_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        collateral_to_receive_usd=Decimal("10500"),
        collateral_asset_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        estimated_gross_profit_usd=Decimal("500"),
        estimated_gas_cost_usd=Decimal("10"),
        estimated_slippage_usd=Decimal("20"),
        estimated_net_profit_usd=Decimal("470"),
        estimated_net_profit_eth=Decimal("0.235"),
    )


@pytest.fixture
def simulation_config() -> SimulationConfig:
    """Create simulation config for testing."""
    return SimulationConfig(
        num_iterations=100,  # Reduced for faster tests
        base_seed=42,
        bootstrap_samples=100,  # Reduced for faster tests
    )


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self) -> None:
        """Should have sensible defaults."""
        config = SimulationConfig()
        assert config.num_iterations == 1000
        assert config.base_seed == 42
        assert config.bootstrap_samples == 1000
        assert config.confidence_level == 0.95

    def test_custom_config(self) -> None:
        """Should accept custom parameters."""
        config = SimulationConfig(
            num_iterations=500,
            base_seed=12345,
            our_execution_latency_ms=50.0,
        )
        assert config.num_iterations == 500
        assert config.base_seed == 12345
        assert config.our_execution_latency_ms == 50.0


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_engine_creation(self, simulation_config: SimulationConfig) -> None:
        """Should create engine with defaults."""
        engine = SimulationEngine(config=simulation_config)
        assert len(engine.bots) >= 10
        assert engine.config.num_iterations == 100

    def test_engine_with_custom_bots(self, simulation_config: SimulationConfig) -> None:
        """Should accept custom bot pool."""
        custom_bots = [
            create_bot(BotType.FRONTRUNNER, "custom_1", seed=1),
            create_bot(BotType.BACKRUNNER, "custom_2", seed=2),
        ]
        engine = SimulationEngine(bots=custom_bots, config=simulation_config)
        assert len(engine.bots) == 2

    def test_simulate_returns_result(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Should return valid SimulationResult."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        assert result.opportunity_id == "test_opp_1"
        assert result.num_iterations == 100
        assert result.random_seed == 42
        assert result.num_competing_bots >= 10

    def test_capture_probability_range(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Capture probability should be between 0 and 1."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        assert Decimal(0) <= result.capture_probability <= Decimal(1)
        assert Decimal(0) <= result.capture_ci_lower_95 <= Decimal(1)
        assert Decimal(0) <= result.capture_ci_upper_95 <= Decimal(1)
        assert result.capture_ci_lower_95 <= result.capture_ci_upper_95

    def test_ev_statistics(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Should calculate EV statistics."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        # Mean should be between min and max
        assert result.min_ev_eth <= result.mean_ev_eth <= result.max_ev_eth

        # CI should bracket the mean
        assert result.ev_ci_lower_95 <= result.mean_ev_eth <= result.ev_ci_upper_95

        # Std should be non-negative
        assert result.std_ev_eth >= 0

    def test_failure_tracking(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Should track failures."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        # Failure count + captures should equal iterations
        capture_count = int(result.capture_probability * result.num_iterations)
        assert result.failure_count + capture_count == result.num_iterations

        # Failure rate should match
        assert result.failure_rate == Decimal(str(result.failure_count / result.num_iterations))

    def test_deterministic_replay(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Same seed should produce same results."""
        engine1 = SimulationEngine(config=simulation_config)
        engine2 = SimulationEngine(config=simulation_config)

        result1 = engine1.simulate(sample_opportunity)
        result2 = engine2.simulate(sample_opportunity)

        assert result1.capture_probability == result2.capture_probability
        assert result1.mean_ev_eth == result2.mean_ev_eth
        assert result1.failure_count == result2.failure_count

    def test_different_seed_different_results(
        self,
        sample_opportunity: LiquidationOpportunity,
    ) -> None:
        """Different seeds should produce different failure distributions."""
        config1 = SimulationConfig(num_iterations=100, base_seed=42)
        config2 = SimulationConfig(num_iterations=100, base_seed=12345)

        engine1 = SimulationEngine(config=config1)
        engine2 = SimulationEngine(config=config2)

        result1 = engine1.simulate(sample_opportunity)
        result2 = engine2.simulate(sample_opportunity)

        # With different seeds, the failure reasons distribution should differ
        # (which bot types win more often)
        assert result1.failure_reasons != result2.failure_reasons


class TestSimulationWithDifferentBotConfigs:
    """Tests for simulation with various bot configurations."""

    def test_no_competition(
        self,
        sample_opportunity: LiquidationOpportunity,
    ) -> None:
        """Should have 100% capture rate with no competing bots."""
        config = SimulationConfig(num_iterations=100, base_seed=42)
        engine = SimulationEngine(bots=[], config=config)  # No competing bots
        result = engine.simulate(sample_opportunity)

        # With no bots, we should capture 100% of the time
        assert result.capture_probability == Decimal(1)
        assert result.num_competing_bots == 0
        assert result.failure_count == 0

    def test_high_competition(
        self,
        sample_opportunity: LiquidationOpportunity,
    ) -> None:
        """Should have lower capture rate with aggressive bots."""
        config = SimulationConfig(num_iterations=100, base_seed=42)

        # Create aggressive frontrunner pool
        aggressive_bots = [
            create_bot(
                BotType.FRONTRUNNER,
                f"aggressive_{i}",
                seed=42 + i,
                activation_probability=0.9,
                execution_latency_ms=30,
            )
            for i in range(5)
        ]

        engine = SimulationEngine(bots=aggressive_bots, config=config)
        result = engine.simulate(sample_opportunity)

        # Should have significant competition
        assert result.capture_probability < Decimal("0.5")
        assert result.failure_count > 50

    def test_gas_sensitive_environment(
        self,
        sample_opportunity: LiquidationOpportunity,
    ) -> None:
        """Gas-sensitive bots should be less active in high gas."""
        # Modify opportunity to have high gas
        high_gas_market = MarketConditions(
            block_number=12345678,
            timestamp=datetime.now(timezone.utc),
            gas_price_gwei=Decimal("2.0"),  # High gas
            eth_price_usd=Decimal("2000"),
        )

        config = SimulationConfig(num_iterations=100, base_seed=42)

        # Only gas-sensitive bots
        gas_bots = [
            create_bot(
                BotType.GAS_SENSITIVE,
                f"gas_{i}",
                seed=42 + i,
                activation_probability=0.8,
                max_gas_gwei=0.5,
            )
            for i in range(5)
        ]

        engine = SimulationEngine(bots=gas_bots, config=config)
        result = engine.simulate(sample_opportunity, market_conditions=high_gas_market)

        # Gas-sensitive bots should not activate much
        assert result.capture_probability > Decimal("0.8")


class TestSimulationBatch:
    """Tests for batch simulation."""

    def test_batch_simulation(self, simulation_config: SimulationConfig) -> None:
        """Should simulate multiple opportunities."""
        opportunities = []
        for i in range(3):
            position = UserPosition(
                user_address=f"0x{i:040x}",
                block_number=12345678 + i,
                collaterals=[],
                debts=[],
            )
            market = MarketConditions(
                block_number=12345678 + i,
                timestamp=datetime.now(timezone.utc),
                gas_price_gwei=Decimal("0.1"),
                eth_price_usd=Decimal("2000"),
            )
            opp = LiquidationOpportunity(
                opportunity_id=f"batch_opp_{i}",
                detected_at_block=12345678 + i,
                detected_at_timestamp=datetime.now(timezone.utc),
                user_address=f"0x{i:040x}",
                position_snapshot=position,
                market_conditions=market,
                debt_to_cover_usd=Decimal("10000"),
                debt_asset_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                collateral_to_receive_usd=Decimal("10500"),
                collateral_asset_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                estimated_gross_profit_usd=Decimal("500"),
                estimated_gas_cost_usd=Decimal("10"),
                estimated_slippage_usd=Decimal("20"),
                estimated_net_profit_usd=Decimal("470"),
                estimated_net_profit_eth=Decimal("0.235"),
            )
            opportunities.append(opp)

        engine = SimulationEngine(config=simulation_config)
        results = run_simulation_batch(opportunities, engine)

        assert len(results) == 3
        assert all(r.num_iterations == 100 for r in results)
        assert results[0].opportunity_id == "batch_opp_0"
        assert results[1].opportunity_id == "batch_opp_1"
        assert results[2].opportunity_id == "batch_opp_2"


class TestSimulationResultProperties:
    """Tests for SimulationResult computed properties."""

    def test_is_profitable(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Should correctly determine profitability."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        # Check profitability logic
        expected_profitable = result.ev_ci_lower_95 > Decimal(0)
        assert result.is_profitable == expected_profitable

    def test_meets_capture_threshold(
        self,
        sample_opportunity: LiquidationOpportunity,
        simulation_config: SimulationConfig,
    ) -> None:
        """Should correctly check capture threshold (3%)."""
        engine = SimulationEngine(config=simulation_config)
        result = engine.simulate(sample_opportunity)

        expected_meets = result.capture_probability >= Decimal("0.03")
        assert result.meets_capture_threshold == expected_meets
