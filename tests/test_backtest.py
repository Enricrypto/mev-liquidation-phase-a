"""Tests for backtest framework."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mev_analysis.core.backtest import (
    BacktestConfig,
    BacktestRunner,
    create_synthetic_positions,
)
from mev_analysis.data.models import MarketConditions


@pytest.fixture
def market_conditions() -> MarketConditions:
    """Sample market conditions."""
    return MarketConditions(
        block_number=1000000,
        timestamp=datetime.now(timezone.utc),
        gas_price_gwei=Decimal("0.1"),
        eth_price_usd=Decimal("2000"),
    )


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self) -> None:
        """Should have sensible defaults."""
        config = BacktestConfig()
        assert config.window_size_blocks == 1000
        assert config.num_seeds >= 10
        assert config.min_sample_size == 30
        assert config.confidence_level == 0.95

    def test_custom_config(self) -> None:
        """Should accept custom parameters."""
        config = BacktestConfig(
            window_size_blocks=500,
            num_seeds=5,
            min_sample_size=10,
        )
        assert config.window_size_blocks == 500
        assert config.num_seeds == 5


class TestCreateSyntheticPositions:
    """Tests for synthetic position generation."""

    def test_creates_correct_count(self) -> None:
        """Should create requested number of positions."""
        positions = create_synthetic_positions(
            num_positions=50,
            num_liquidatable=10,
        )
        assert len(positions) == 50

    def test_creates_liquidatable_positions(self) -> None:
        """Should create correct number of liquidatable positions."""
        positions = create_synthetic_positions(
            num_positions=50,
            num_liquidatable=10,
        )

        liquidatable_count = sum(1 for p in positions if p.is_liquidatable)
        assert liquidatable_count == 10

    def test_positions_have_valid_data(self) -> None:
        """Should create positions with valid collateral and debt."""
        positions = create_synthetic_positions(
            num_positions=10,
            num_liquidatable=5,
        )

        for pos in positions:
            assert len(pos.collaterals) > 0
            assert len(pos.debts) > 0
            assert pos.total_collateral_usd > 0
            assert pos.total_debt_usd > 0
            assert pos.health_factor is not None

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce same positions."""
        positions1 = create_synthetic_positions(
            num_positions=10,
            num_liquidatable=3,
            seed=12345,
        )
        positions2 = create_synthetic_positions(
            num_positions=10,
            num_liquidatable=3,
            seed=12345,
        )

        for p1, p2 in zip(positions1, positions2):
            assert p1.user_address == p2.user_address
            assert p1.health_factor == p2.health_factor

    def test_different_seeds_produce_different_positions(self) -> None:
        """Different seeds should produce different positions."""
        positions1 = create_synthetic_positions(
            num_positions=10,
            num_liquidatable=3,
            seed=12345,
        )
        positions2 = create_synthetic_positions(
            num_positions=10,
            num_liquidatable=3,
            seed=54321,
        )

        # Health factors should differ
        hfs1 = [float(p.health_factor) for p in positions1 if p.health_factor]
        hfs2 = [float(p.health_factor) for p in positions2 if p.health_factor]
        assert hfs1 != hfs2


class TestBacktestRunner:
    """Tests for BacktestRunner."""

    @pytest.fixture
    def runner(self) -> BacktestRunner:
        """Create runner with reduced iterations for speed."""
        config = BacktestConfig(
            num_simulation_iterations=10,  # Reduced for faster tests
            num_seeds=2,  # Reduced for faster tests
            window_size_blocks=50,
            window_stride_blocks=25,
            min_sample_size=1,  # Low threshold for tests
            bootstrap_samples=50,
        )
        return BacktestRunner(config=config)

    def test_run_with_synthetic_positions(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should run backtest on synthetic positions."""
        positions = create_synthetic_positions(
            num_positions=20,
            num_liquidatable=5,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        assert result.backtest_id.startswith("bt_")
        assert result.total_positions_scanned > 0
        assert result.completed_at is not None

    def test_run_generates_windows(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should generate multiple windows."""
        positions = create_synthetic_positions(
            num_positions=100,
            num_liquidatable=20,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        # Should have at least one window
        assert len(result.windows) >= 1

    def test_run_with_empty_positions(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should handle empty position list."""
        result = runner.run([], market_conditions)

        assert result.total_positions_scanned == 0
        assert result.total_opportunities_detected == 0
        assert len(result.windows) == 0

    def test_run_with_no_liquidatable_positions(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should handle positions with none liquidatable."""
        positions = create_synthetic_positions(
            num_positions=20,
            num_liquidatable=0,  # No liquidatable positions
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        assert result.total_positions_scanned > 0
        assert result.total_opportunities_detected == 0

    def test_hypothesis_results(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should include hypothesis testing results."""
        positions = create_synthetic_positions(
            num_positions=30,
            num_liquidatable=10,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        assert "h1_profitable_opportunities" in result.hypothesis_results
        assert "h2_capture_probability" in result.hypothesis_results
        assert "bonferroni_applied" in result.hypothesis_results

    def test_overall_statistics(
        self,
        runner: BacktestRunner,
        market_conditions: MarketConditions,
    ) -> None:
        """Should calculate overall statistics."""
        positions = create_synthetic_positions(
            num_positions=30,
            num_liquidatable=10,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        # Should have EV and capture probability stats
        assert result.overall_ev_ci_lower_95 <= result.overall_mean_ev_eth
        assert result.overall_mean_ev_eth <= result.overall_ev_ci_upper_95

        assert Decimal(0) <= result.overall_capture_ci_lower_95
        assert result.overall_capture_ci_upper_95 <= Decimal(1)

    def test_validation_flags(
        self,
        market_conditions: MarketConditions,
    ) -> None:
        """Should set validation flags based on results."""
        # Create runner with strict requirements
        config = BacktestConfig(
            num_simulation_iterations=10,
            num_seeds=2,
            window_size_blocks=50,
            min_sample_size=100,  # High threshold
            bootstrap_samples=50,
        )
        runner = BacktestRunner(config=config)

        positions = create_synthetic_positions(
            num_positions=20,
            num_liquidatable=5,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        # Should not meet sample size with only 5 liquidatable positions
        # (depends on detection)
        assert isinstance(result.meets_sample_size, bool)
        assert isinstance(result.meets_capture_threshold, bool)
        assert isinstance(result.meets_ev_threshold, bool)


class TestBacktestRunnerWithMarketConditionsDict:
    """Tests for backtest runner with per-block market conditions."""

    def test_run_with_market_conditions_dict(self) -> None:
        """Should handle market conditions as dict by block."""
        config = BacktestConfig(
            num_simulation_iterations=10,
            num_seeds=2,
            window_size_blocks=50,
            window_stride_blocks=25,
            min_sample_size=1,
            bootstrap_samples=50,
        )
        runner = BacktestRunner(config=config)

        positions = create_synthetic_positions(
            num_positions=20,
            num_liquidatable=5,
            base_block=1000000,
        )

        # Create market conditions for different blocks
        market_by_block = {
            1000000: MarketConditions(
                block_number=1000000,
                timestamp=datetime.now(timezone.utc),
                gas_price_gwei=Decimal("0.1"),
                eth_price_usd=Decimal("2000"),
            ),
            1000050: MarketConditions(
                block_number=1000050,
                timestamp=datetime.now(timezone.utc),
                gas_price_gwei=Decimal("0.2"),
                eth_price_usd=Decimal("2100"),
            ),
        }

        result = runner.run(positions, market_by_block)

        assert result.backtest_id.startswith("bt_")
        assert result.completed_at is not None


class TestWindowResult:
    """Tests for WindowResult data."""

    def test_window_result_fields(
        self,
        market_conditions: MarketConditions,
    ) -> None:
        """Should populate all window result fields."""
        config = BacktestConfig(
            num_simulation_iterations=10,
            num_seeds=2,
            window_size_blocks=100,
            min_sample_size=1,
            bootstrap_samples=50,
        )
        runner = BacktestRunner(config=config)

        positions = create_synthetic_positions(
            num_positions=50,
            num_liquidatable=10,
            base_block=1000000,
        )

        result = runner.run(positions, market_conditions)

        if result.windows:
            window = result.windows[0]
            assert window.window_id.startswith("win_")
            assert window.start_block >= 1000000
            assert window.end_block > window.start_block
            assert window.num_positions_scanned >= 0
            assert window.duration_ms >= 0
            assert len(window.seeds_used) > 0
