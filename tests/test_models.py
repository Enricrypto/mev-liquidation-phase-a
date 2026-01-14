"""Tests for data models."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mev_analysis.data.models import (
    Asset,
    AssetType,
    CollateralAsset,
    DebtAsset,
    KnownPosition,
    LiquidationOpportunity,
    MarketConditions,
    PositionStatus,
    SimulationResult,
    UserPosition,
)


class TestAsset:
    """Tests for Asset model."""

    def test_asset_creation(self) -> None:
        """Should create asset with correct values."""
        asset = Asset(
            symbol="WETH",
            address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            decimals=18,
            amount_raw=1_000_000_000_000_000_000,  # 1 ETH
            price_usd=Decimal("2000.00"),
            asset_type=AssetType.ETH_CORRELATED,
        )
        assert asset.symbol == "WETH"
        assert asset.amount == Decimal(1)
        assert asset.value_usd == Decimal("2000.00")

    def test_asset_computed_properties(self) -> None:
        """Should correctly compute amount and value."""
        asset = Asset(
            symbol="USDC",
            address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            decimals=6,
            amount_raw=1_500_000_000,  # 1500 USDC
            price_usd=Decimal("1.00"),
            asset_type=AssetType.STABLE,
        )
        assert asset.amount == Decimal("1500")
        assert asset.value_usd == Decimal("1500.00")


class TestCollateralAsset:
    """Tests for CollateralAsset model."""

    def test_collateral_with_threshold(self) -> None:
        """Should calculate liquidation value correctly."""
        collateral = CollateralAsset(
            symbol="WETH",
            address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            decimals=18,
            amount_raw=2_000_000_000_000_000_000,  # 2 ETH
            price_usd=Decimal("2000.00"),
            asset_type=AssetType.ETH_CORRELATED,
            liquidation_threshold=Decimal("0.825"),
            liquidation_bonus=Decimal("0.05"),
            ltv=Decimal("0.80"),
        )
        assert collateral.value_usd == Decimal("4000.00")
        assert collateral.liquidation_value_usd == Decimal("3300.00")  # 4000 * 0.825

    def test_collateral_disabled(self) -> None:
        """Should return zero liquidation value when disabled."""
        collateral = CollateralAsset(
            symbol="ARB",
            address="0x912CE59144191C1204E64559FE8253a0e49E6548",
            decimals=18,
            amount_raw=1000_000_000_000_000_000_000,  # 1000 ARB
            price_usd=Decimal("1.50"),
            asset_type=AssetType.VOLATILE,
            liquidation_threshold=Decimal("0.70"),
            liquidation_bonus=Decimal("0.10"),
            ltv=Decimal("0.65"),
            usage_as_collateral_enabled=False,
        )
        assert collateral.value_usd == Decimal("1500.00")
        assert collateral.liquidation_value_usd == Decimal(0)


class TestUserPosition:
    """Tests for UserPosition model."""

    @pytest.fixture
    def sample_position(self) -> UserPosition:
        """Create a sample user position."""
        return UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=2_000_000_000_000_000_000,  # 2 ETH
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.825"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.80"),
                ),
            ],
            debts=[
                DebtAsset(
                    symbol="USDC",
                    address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    decimals=6,
                    amount_raw=3_000_000_000,  # 3000 USDC
                    price_usd=Decimal("1.00"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
        )

    def test_position_address_normalization(self) -> None:
        """Should normalize address to lowercase."""
        position = UserPosition(
            user_address="0xABCDEF1234567890ABCDEF1234567890ABCDEF12",
            block_number=1,
            collaterals=[],
            debts=[],
        )
        assert position.user_address == "0xabcdef1234567890abcdef1234567890abcdef12"

    def test_position_totals(self, sample_position: UserPosition) -> None:
        """Should calculate totals correctly."""
        assert sample_position.total_collateral_usd == Decimal("4000.00")
        assert sample_position.total_collateral_liquidation_usd == Decimal("3300.00")
        assert sample_position.total_debt_usd == Decimal("3000.00")

    def test_calculated_health_factor(self, sample_position: UserPosition) -> None:
        """Should calculate health factor correctly."""
        # HF = 3300 / 3000 = 1.1
        hf = sample_position.calculated_health_factor
        assert hf is not None
        assert hf == Decimal("1.1")

    def test_health_factor_no_debt(self) -> None:
        """Should return None for positions with no debt."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=1,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=1_000_000_000_000_000_000,
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.825"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.80"),
                ),
            ],
            debts=[],
        )
        assert position.calculated_health_factor is None
        assert position.status == PositionStatus.HEALTHY

    def test_position_status_healthy(self, sample_position: UserPosition) -> None:
        """Should classify healthy positions correctly."""
        sample_position.health_factor = Decimal("1.5")
        assert sample_position.status == PositionStatus.HEALTHY
        assert not sample_position.is_liquidatable

    def test_position_status_at_risk(self) -> None:
        """Should classify at-risk positions correctly."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=1,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=1_000_000_000_000_000_000,
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.825"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.80"),
                ),
            ],
            debts=[
                DebtAsset(
                    symbol="USDC",
                    address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    decimals=6,
                    amount_raw=1_400_000_000,  # 1400 USDC
                    price_usd=Decimal("1.00"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
            health_factor=Decimal("1.2"),  # Between 1.0 and 1.5
        )
        assert position.status == PositionStatus.AT_RISK
        assert not position.is_liquidatable

    def test_position_status_liquidatable(self) -> None:
        """Should classify liquidatable positions correctly."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=1,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=1_000_000_000_000_000_000,
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.825"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.80"),
                ),
            ],
            debts=[
                DebtAsset(
                    symbol="USDC",
                    address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    decimals=6,
                    amount_raw=2_000_000_000,  # 2000 USDC
                    price_usd=Decimal("1.00"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
            health_factor=Decimal("0.825"),  # Below 1.0
        )
        assert position.status == PositionStatus.LIQUIDATABLE
        assert position.is_liquidatable


class TestMarketConditions:
    """Tests for MarketConditions model."""

    def test_market_conditions_creation(self) -> None:
        """Should create market conditions."""
        conditions = MarketConditions(
            block_number=12345678,
            timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
            gas_price_gwei=Decimal("0.1"),
            base_fee_gwei=Decimal("0.05"),
            eth_price_usd=Decimal("2500.00"),
        )
        assert conditions.hour_of_day == 14
        assert conditions.day_of_week == 0  # Monday
        assert not conditions.is_weekend

    def test_weekend_detection(self) -> None:
        """Should detect weekends correctly."""
        saturday = MarketConditions(
            block_number=1,
            timestamp=datetime(2024, 1, 13, 12, 0, 0, tzinfo=timezone.utc),  # Saturday
            gas_price_gwei=Decimal("0.1"),
            eth_price_usd=Decimal("2000.00"),
        )
        assert saturday.is_weekend


class TestKnownPosition:
    """Tests for KnownPosition model."""

    def test_address_normalization(self) -> None:
        """Should normalize address."""
        pos = KnownPosition(user_address="ABCDEF1234567890ABCDEF1234567890ABCDEF12")
        assert pos.user_address == "0xabcdef1234567890abcdef1234567890abcdef12"

    def test_with_block_number(self) -> None:
        """Should accept block number."""
        pos = KnownPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            label="test_whale",
            source="historical",
        )
        assert pos.block_number == 12345678
        assert pos.label == "test_whale"


class TestLiquidationOpportunity:
    """Tests for LiquidationOpportunity model."""

    def test_actionable_threshold(self) -> None:
        """Should correctly determine if opportunity is actionable."""
        # Create minimal position for the snapshot
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
            eth_price_usd=Decimal("2000.00"),
        )

        # Opportunity with 0.015 ETH profit (above 0.01 threshold)
        opportunity = LiquidationOpportunity(
            opportunity_id="test_1",
            detected_at_block=12345678,
            detected_at_timestamp=datetime.now(timezone.utc),
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            position_snapshot=position,
            market_conditions=market,
            debt_to_cover_usd=Decimal("1000.00"),
            debt_asset_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            collateral_to_receive_usd=Decimal("1050.00"),
            collateral_asset_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            estimated_gross_profit_usd=Decimal("50.00"),
            estimated_gas_cost_usd=Decimal("5.00"),
            estimated_slippage_usd=Decimal("5.00"),
            estimated_net_profit_usd=Decimal("40.00"),
            estimated_net_profit_eth=Decimal("0.015"),
        )
        assert opportunity.is_actionable

        # Opportunity with 0.005 ETH profit (below threshold)
        opportunity_small = LiquidationOpportunity(
            opportunity_id="test_2",
            detected_at_block=12345678,
            detected_at_timestamp=datetime.now(timezone.utc),
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            position_snapshot=position,
            market_conditions=market,
            debt_to_cover_usd=Decimal("100.00"),
            debt_asset_address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            collateral_to_receive_usd=Decimal("105.00"),
            collateral_asset_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            estimated_gross_profit_usd=Decimal("5.00"),
            estimated_gas_cost_usd=Decimal("3.00"),
            estimated_slippage_usd=Decimal("1.00"),
            estimated_net_profit_usd=Decimal("1.00"),
            estimated_net_profit_eth=Decimal("0.005"),
        )
        assert not opportunity_small.is_actionable


class TestSimulationResult:
    """Tests for SimulationResult model."""

    def test_profitability_check(self) -> None:
        """Should check profitability with CI."""
        # Profitable result (CI lower bound > 0)
        profitable = SimulationResult(
            opportunity_id="test_1",
            simulation_id="sim_1",
            num_iterations=1000,
            random_seed=42,
            num_competing_bots=10,
            mean_ev_eth=Decimal("0.05"),
            std_ev_eth=Decimal("0.01"),
            min_ev_eth=Decimal("0.02"),
            max_ev_eth=Decimal("0.08"),
            ev_ci_lower_95=Decimal("0.03"),
            ev_ci_upper_95=Decimal("0.07"),
            capture_probability=Decimal("0.15"),
            capture_ci_lower_95=Decimal("0.10"),
            capture_ci_upper_95=Decimal("0.20"),
            mean_gas_cost_eth=Decimal("0.001"),
            mean_slippage_eth=Decimal("0.002"),
            failure_count=50,
            failure_rate=Decimal("0.05"),
            simulation_duration_ms=500,
        )
        assert profitable.is_profitable
        assert profitable.meets_capture_threshold

    def test_unprofitable_result(self) -> None:
        """Should identify unprofitable results."""
        unprofitable = SimulationResult(
            opportunity_id="test_2",
            simulation_id="sim_2",
            num_iterations=1000,
            random_seed=42,
            num_competing_bots=10,
            mean_ev_eth=Decimal("0.001"),
            std_ev_eth=Decimal("0.005"),
            min_ev_eth=Decimal("-0.01"),
            max_ev_eth=Decimal("0.02"),
            ev_ci_lower_95=Decimal("-0.002"),  # Lower bound negative
            ev_ci_upper_95=Decimal("0.004"),
            capture_probability=Decimal("0.02"),  # Below 3% threshold
            capture_ci_lower_95=Decimal("0.01"),
            capture_ci_upper_95=Decimal("0.03"),
            mean_gas_cost_eth=Decimal("0.001"),
            mean_slippage_eth=Decimal("0.002"),
            failure_count=200,
            failure_rate=Decimal("0.20"),
            simulation_duration_ms=500,
        )
        assert not unprofitable.is_profitable
        assert not unprofitable.meets_capture_threshold
