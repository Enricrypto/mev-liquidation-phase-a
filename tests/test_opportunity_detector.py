"""Tests for opportunity detector."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mev_analysis.core.opportunity_detector import (
    DetectorConfig,
    OpportunityDetector,
    filter_actionable_opportunities,
)
from mev_analysis.data.models import (
    AssetType,
    CollateralAsset,
    DebtAsset,
    LiquidationOpportunity,
    MarketConditions,
    UserPosition,
)


@pytest.fixture
def market_conditions() -> MarketConditions:
    """Sample market conditions."""
    return MarketConditions(
        block_number=12345678,
        timestamp=datetime.now(timezone.utc),
        gas_price_gwei=Decimal("0.1"),
        eth_price_usd=Decimal("2000"),
    )


@pytest.fixture
def liquidatable_position() -> UserPosition:
    """Create a liquidatable position (HF < 1)."""
    return UserPosition(
        user_address="0x1234567890abcdef1234567890abcdef12345678",
        block_number=12345678,
        collaterals=[
            CollateralAsset(
                symbol="WETH",
                address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                decimals=18,
                amount_raw=1_000_000_000_000_000_000,  # 1 ETH
                price_usd=Decimal("2000"),
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
                amount_raw=1_800_000_000,  # 1800 USDC
                price_usd=Decimal("1.0"),
                asset_type=AssetType.STABLE,
                is_stable_rate=False,
                current_rate=Decimal("0.05"),
            ),
        ],
        health_factor=Decimal("0.92"),  # Below 1.0
        health_factor_source="on_chain",
    )


@pytest.fixture
def healthy_position() -> UserPosition:
    """Create a healthy position (HF > 1)."""
    return UserPosition(
        user_address="0xabcdef1234567890abcdef1234567890abcdef12",
        block_number=12345678,
        collaterals=[
            CollateralAsset(
                symbol="WETH",
                address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                decimals=18,
                amount_raw=1_000_000_000_000_000_000,
                price_usd=Decimal("2000"),
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
                amount_raw=1_000_000_000,  # 1000 USDC
                price_usd=Decimal("1.0"),
                asset_type=AssetType.STABLE,
                is_stable_rate=False,
                current_rate=Decimal("0.05"),
            ),
        ],
        health_factor=Decimal("1.65"),  # Above 1.0
        health_factor_source="on_chain",
    )


class TestDetectorConfig:
    """Tests for DetectorConfig."""

    def test_default_config(self) -> None:
        """Should have sensible defaults."""
        config = DetectorConfig()
        assert config.min_profit_eth == Decimal("0.01")
        assert config.min_debt_usd == Decimal("100")
        assert config.estimated_gas_units == 350_000

    def test_custom_config(self) -> None:
        """Should accept custom parameters."""
        config = DetectorConfig(
            min_profit_eth=Decimal("0.05"),
            min_debt_usd=Decimal("500"),
        )
        assert config.min_profit_eth == Decimal("0.05")
        assert config.min_debt_usd == Decimal("500")


class TestOpportunityDetector:
    """Tests for OpportunityDetector."""

    def test_detect_liquidatable_position(
        self,
        liquidatable_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should detect liquidatable position as opportunity."""
        detector = OpportunityDetector()
        opportunity = detector.detect_single(liquidatable_position, market_conditions)

        assert opportunity is not None
        assert opportunity.user_address == liquidatable_position.user_address
        assert opportunity.estimated_net_profit_eth > 0

    def test_skip_healthy_position(
        self,
        healthy_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should not detect healthy position as opportunity."""
        detector = OpportunityDetector()
        opportunity = detector.detect_single(healthy_position, market_conditions)

        assert opportunity is None

    def test_detect_multiple_positions(
        self,
        liquidatable_position: UserPosition,
        healthy_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should detect only liquidatable positions from list."""
        detector = OpportunityDetector()
        positions = [liquidatable_position, healthy_position]

        opportunities = list(detector.detect(positions, market_conditions))

        assert len(opportunities) == 1
        assert opportunities[0].user_address == liquidatable_position.user_address

    def test_skip_small_positions(
        self,
        market_conditions: MarketConditions,
    ) -> None:
        """Should skip positions with debt below minimum."""
        small_position = UserPosition(
            user_address="0x0000000000000000000000000000000000000001",
            block_number=12345678,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=10_000_000_000_000_000,  # 0.01 ETH
                    price_usd=Decimal("2000"),
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
                    amount_raw=50_000_000,  # 50 USDC (below 100 minimum)
                    price_usd=Decimal("1.0"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
            health_factor=Decimal("0.8"),
        )

        config = DetectorConfig(min_debt_usd=Decimal("100"))
        detector = OpportunityDetector(config)
        opportunity = detector.detect_single(small_position, market_conditions)

        assert opportunity is None

    def test_close_factor_50_percent(
        self,
        liquidatable_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should use 50% close factor for HF between 0.95 and 1.0."""
        # HF = 0.97 (between 0.95 and 1.0)
        liquidatable_position.health_factor = Decimal("0.97")

        detector = OpportunityDetector()
        opportunity = detector.detect_single(liquidatable_position, market_conditions)

        assert opportunity is not None
        # Should be around 50% of debt
        max_possible_debt = liquidatable_position.total_debt_usd
        assert opportunity.debt_to_cover_usd <= max_possible_debt * Decimal("0.51")

    def test_close_factor_100_percent(
        self,
        market_conditions: MarketConditions,
    ) -> None:
        """Should use 100% close factor for HF below 0.95."""
        position = UserPosition(
            user_address="0x0000000000000000000000000000000000000002",
            block_number=12345678,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=5_000_000_000_000_000_000,  # 5 ETH
                    price_usd=Decimal("2000"),
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
                    amount_raw=10_000_000_000,  # 10000 USDC
                    price_usd=Decimal("1.0"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
            health_factor=Decimal("0.80"),  # Below 0.95
        )

        detector = OpportunityDetector()
        opportunity = detector.detect_single(position, market_conditions)

        assert opportunity is not None
        # Should allow up to 100% of debt
        assert opportunity.debt_to_cover_usd <= position.total_debt_usd

    def test_opportunity_includes_gas_estimate(
        self,
        liquidatable_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should include gas cost estimate."""
        detector = OpportunityDetector()
        opportunity = detector.detect_single(liquidatable_position, market_conditions)

        assert opportunity is not None
        assert opportunity.estimated_gas_cost_usd > 0

    def test_opportunity_includes_slippage_estimate(
        self,
        liquidatable_position: UserPosition,
        market_conditions: MarketConditions,
    ) -> None:
        """Should include slippage estimate."""
        detector = OpportunityDetector()
        opportunity = detector.detect_single(liquidatable_position, market_conditions)

        assert opportunity is not None
        assert opportunity.estimated_slippage_usd >= 0


class TestFilterActionableOpportunities:
    """Tests for filter_actionable_opportunities."""

    def test_filter_by_profit(self) -> None:
        """Should filter out low-profit opportunities."""
        # Create mock opportunities with different profits
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

        high_profit = LiquidationOpportunity(
            opportunity_id="high",
            detected_at_block=12345678,
            detected_at_timestamp=datetime.now(timezone.utc),
            user_address="0x1",
            position_snapshot=position,
            market_conditions=market,
            debt_to_cover_usd=Decimal("1000"),
            debt_asset_address="0x1",
            collateral_to_receive_usd=Decimal("1050"),
            collateral_asset_address="0x2",
            estimated_gross_profit_usd=Decimal("50"),
            estimated_gas_cost_usd=Decimal("5"),
            estimated_slippage_usd=Decimal("5"),
            estimated_net_profit_usd=Decimal("40"),
            estimated_net_profit_eth=Decimal("0.02"),  # Above threshold
        )

        low_profit = LiquidationOpportunity(
            opportunity_id="low",
            detected_at_block=12345678,
            detected_at_timestamp=datetime.now(timezone.utc),
            user_address="0x2",
            position_snapshot=position,
            market_conditions=market,
            debt_to_cover_usd=Decimal("100"),
            debt_asset_address="0x1",
            collateral_to_receive_usd=Decimal("105"),
            collateral_asset_address="0x2",
            estimated_gross_profit_usd=Decimal("5"),
            estimated_gas_cost_usd=Decimal("4"),
            estimated_slippage_usd=Decimal("1"),
            estimated_net_profit_usd=Decimal("0"),
            estimated_net_profit_eth=Decimal("0.005"),  # Below threshold
        )

        filtered = filter_actionable_opportunities(
            [high_profit, low_profit],
            min_profit_eth=Decimal("0.01"),
        )

        assert len(filtered) == 1
        assert filtered[0].opportunity_id == "high"
