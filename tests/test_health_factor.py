"""Tests for health factor calculation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from mev_analysis.data.models import (
    AssetType,
    CollateralAsset,
    DebtAsset,
    UserPosition,
)
from mev_analysis.data.health_factor import HealthFactorCalculator


class TestHealthFactorCalculator:
    """Tests for HealthFactorCalculator."""

    @pytest.fixture
    def calculator(self) -> HealthFactorCalculator:
        """Create calculator instance."""
        return HealthFactorCalculator(verification_tolerance=Decimal("0.001"))

    @pytest.fixture
    def sample_collaterals(self) -> list[CollateralAsset]:
        """Sample collateral assets."""
        return [
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
        ]

    @pytest.fixture
    def sample_debts(self) -> list[DebtAsset]:
        """Sample debt assets."""
        return [
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
        ]

    def test_calculate_health_factor(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should calculate health factor correctly."""
        # Collateral: 2 ETH * $2000 * 0.825 = $3300
        # Debt: 3000 USDC
        # HF = 3300 / 3000 = 1.1
        hf = calculator.calculate_health_factor(sample_collaterals, sample_debts)
        assert hf == Decimal("1.1")

    def test_health_factor_no_debt(
        self, calculator: HealthFactorCalculator, sample_collaterals: list[CollateralAsset]
    ) -> None:
        """Should return None when no debt."""
        hf = calculator.calculate_health_factor(sample_collaterals, [])
        assert hf is None

    def test_health_factor_multiple_collaterals(
        self, calculator: HealthFactorCalculator
    ) -> None:
        """Should sum multiple collaterals correctly."""
        collaterals = [
            CollateralAsset(
                symbol="WETH",
                address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                decimals=18,
                amount_raw=1_000_000_000_000_000_000,  # 1 ETH
                price_usd=Decimal("2000.00"),
                asset_type=AssetType.ETH_CORRELATED,
                liquidation_threshold=Decimal("0.825"),
                liquidation_bonus=Decimal("0.05"),
                ltv=Decimal("0.80"),
            ),
            CollateralAsset(
                symbol="WBTC",
                address="0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
                decimals=8,
                amount_raw=10_000_000,  # 0.1 BTC
                price_usd=Decimal("40000.00"),
                asset_type=AssetType.BTC_CORRELATED,
                liquidation_threshold=Decimal("0.75"),
                liquidation_bonus=Decimal("0.05"),
                ltv=Decimal("0.70"),
            ),
        ]
        debts = [
            DebtAsset(
                symbol="USDC",
                address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                decimals=6,
                amount_raw=4_000_000_000,  # 4000 USDC
                price_usd=Decimal("1.00"),
                asset_type=AssetType.STABLE,
                is_stable_rate=False,
                current_rate=Decimal("0.05"),
            ),
        ]

        # WETH: 1 * 2000 * 0.825 = 1650
        # WBTC: 0.1 * 40000 * 0.75 = 3000
        # Total liquidation collateral: 4650
        # HF = 4650 / 4000 = 1.1625
        hf = calculator.calculate_health_factor(collaterals, debts)
        assert hf == Decimal("1.1625")

    def test_calculate_with_breakdown(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should provide detailed breakdown."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=sample_collaterals,
            debts=sample_debts,
            health_factor=Decimal("1.1"),  # Match calculated
            health_factor_source="on_chain",
        )

        breakdown = calculator.calculate_with_breakdown(position)

        assert breakdown.total_collateral_usd == Decimal("4000.00")
        assert breakdown.total_collateral_liquidation_usd == Decimal("3300.00")
        assert breakdown.total_debt_usd == Decimal("3000.00")
        assert breakdown.calculated_health_factor == Decimal("1.1")
        assert breakdown.verification_passed

    def test_verification_failure(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should detect verification failures."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=sample_collaterals,
            debts=sample_debts,
            health_factor=Decimal("1.5"),  # Different from calculated 1.1
            health_factor_source="on_chain",
        )

        breakdown = calculator.calculate_with_breakdown(position)

        assert not breakdown.verification_passed
        assert breakdown.verification_diff is not None
        # Calculated: 1.1, On-chain: 1.5, Diff: -0.4
        assert breakdown.verification_diff == Decimal("-0.4")

    def test_simulate_price_change_collateral_drop(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should simulate collateral price drop."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=sample_collaterals,
            debts=sample_debts,
        )

        # Original HF = 1.1
        # 20% price drop: 0.8 * 4000 * 0.825 / 3000 = 0.88
        new_hf = calculator.simulate_price_change(
            position,
            "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            Decimal("-20"),
        )

        assert new_hf is not None
        assert new_hf == Decimal("0.88")
        assert new_hf < Decimal(1)  # Now liquidatable

    def test_simulate_price_change_debt_increase(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should simulate debt asset price increase."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=sample_collaterals,
            debts=sample_debts,
        )

        # If USDC price increases (depegging scenario)
        # 10% increase: 3300 / (3000 * 1.1) = 1.0
        new_hf = calculator.simulate_price_change(
            position,
            "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
            Decimal("10"),
        )

        assert new_hf is not None
        assert new_hf == Decimal(1)  # Exactly at liquidation threshold

    def test_find_liquidation_price_collateral(
        self, calculator: HealthFactorCalculator
    ) -> None:
        """Should find price that triggers liquidation."""
        collaterals = [
            CollateralAsset(
                symbol="WETH",
                address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                decimals=18,
                amount_raw=1_000_000_000_000_000_000,  # 1 ETH
                price_usd=Decimal("2000.00"),
                asset_type=AssetType.ETH_CORRELATED,
                liquidation_threshold=Decimal("0.80"),
                liquidation_bonus=Decimal("0.05"),
                ltv=Decimal("0.75"),
            ),
        ]
        debts = [
            DebtAsset(
                symbol="USDC",
                address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                decimals=6,
                amount_raw=1_200_000_000,  # 1200 USDC
                price_usd=Decimal("1.00"),
                asset_type=AssetType.STABLE,
                is_stable_rate=False,
                current_rate=Decimal("0.05"),
            ),
        ]

        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=collaterals,
            debts=debts,
        )

        # Current HF = (2000 * 0.80) / 1200 = 1.333...
        # Liquidation when HF < 1: price * 0.80 / 1200 < 1
        # price < 1500, so need ~25% drop

        liq_price = calculator.find_liquidation_price(
            position,
            "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
            is_collateral=True,
            precision=Decimal("1"),
        )

        assert liq_price is not None
        # Should be around -25%
        assert liq_price < Decimal("-20")
        assert liq_price > Decimal("-30")

    def test_calculate_max_liquidation_healthy(
        self,
        calculator: HealthFactorCalculator,
        sample_collaterals: list[CollateralAsset],
        sample_debts: list[DebtAsset],
    ) -> None:
        """Should return not liquidatable for healthy positions."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=sample_collaterals,
            debts=sample_debts,
            health_factor=Decimal("1.1"),
        )

        result = calculator.calculate_max_liquidation(position)
        assert not result["is_liquidatable"]

    def test_calculate_max_liquidation_partial(
        self, calculator: HealthFactorCalculator
    ) -> None:
        """Should calculate 50% liquidation for HF between 0.95 and 1."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=1_000_000_000_000_000_000,
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.80"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.75"),
                ),
            ],
            debts=[
                DebtAsset(
                    symbol="USDC",
                    address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
                    decimals=6,
                    amount_raw=1_700_000_000,  # 1700 USDC
                    price_usd=Decimal("1.00"),
                    asset_type=AssetType.STABLE,
                    is_stable_rate=False,
                    current_rate=Decimal("0.05"),
                ),
            ],
            health_factor=Decimal("0.97"),  # Between 0.95 and 1.0
        )

        result = calculator.calculate_max_liquidation(position)

        assert result["is_liquidatable"]
        assert result["close_factor"] == "0.5"
        assert Decimal(result["max_debt_to_cover_usd"]) == Decimal("850")

    def test_calculate_max_liquidation_full(
        self, calculator: HealthFactorCalculator
    ) -> None:
        """Should calculate 100% liquidation for HF below 0.95."""
        position = UserPosition(
            user_address="0x1234567890abcdef1234567890abcdef12345678",
            block_number=12345678,
            collaterals=[
                CollateralAsset(
                    symbol="WETH",
                    address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                    decimals=18,
                    amount_raw=1_000_000_000_000_000_000,
                    price_usd=Decimal("2000.00"),
                    asset_type=AssetType.ETH_CORRELATED,
                    liquidation_threshold=Decimal("0.80"),
                    liquidation_bonus=Decimal("0.05"),
                    ltv=Decimal("0.75"),
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
            health_factor=Decimal("0.8"),  # Below 0.95
        )

        result = calculator.calculate_max_liquidation(position)

        assert result["is_liquidatable"]
        assert result["close_factor"] == "1"
        assert Decimal(result["max_debt_to_cover_usd"]) == Decimal("2000")
