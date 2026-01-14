"""Health factor calculation and verification.

Implements Aave v3 health factor calculation logic for:
- Independent verification of on-chain values
- Simulation of health factor changes under stress
- E-mode support

Health Factor = Sum(Collateral_i * Price_i * LiquidationThreshold_i) / TotalDebt
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from mev_analysis.data.models import CollateralAsset, DebtAsset, UserPosition


@dataclass
class HealthFactorBreakdown:
    """Detailed breakdown of health factor calculation."""

    total_collateral_usd: Decimal
    total_collateral_liquidation_usd: Decimal  # Adjusted by thresholds
    total_debt_usd: Decimal
    calculated_health_factor: Decimal | None
    on_chain_health_factor: Decimal | None

    # Per-asset contributions
    collateral_contributions: list[dict[str, Any]]
    debt_contributions: list[dict[str, Any]]

    # Verification
    verification_passed: bool
    verification_diff: Decimal | None
    verification_tolerance: Decimal = Decimal("0.001")  # 0.1% tolerance

    @property
    def diff_pct(self) -> Decimal | None:
        """Percentage difference between calculated and on-chain."""
        if self.verification_diff is None or self.on_chain_health_factor is None:
            return None
        if self.on_chain_health_factor == 0:
            return None
        return abs(self.verification_diff / self.on_chain_health_factor) * 100


class HealthFactorCalculator:
    """Calculate and verify health factors for Aave v3 positions.

    Implements the standard Aave v3 health factor formula with support for:
    - Multiple collateral assets with different thresholds
    - E-mode categories (simplified)
    - Verification against on-chain values

    Usage:
        calculator = HealthFactorCalculator()
        breakdown = calculator.calculate_with_breakdown(position)
        if not breakdown.verification_passed:
            print(f"Verification failed: diff={breakdown.verification_diff}")
    """

    def __init__(
        self,
        verification_tolerance: Decimal = Decimal("0.001"),
    ) -> None:
        """Initialize calculator.

        Args:
            verification_tolerance: Maximum allowed difference between
                calculated and on-chain health factor (as ratio, e.g., 0.001 = 0.1%).
        """
        self.verification_tolerance = verification_tolerance

    def calculate_health_factor(
        self,
        collaterals: list[CollateralAsset],
        debts: list[DebtAsset],
    ) -> Decimal | None:
        """Calculate health factor from assets.

        Args:
            collaterals: List of collateral assets.
            debts: List of debt assets.

        Returns:
            Health factor as Decimal, or None if no debt.
        """
        # Calculate total debt
        total_debt = sum((d.value_usd for d in debts), Decimal(0))

        if total_debt == 0:
            return None  # Infinite health factor (no debt)

        # Calculate liquidation-adjusted collateral
        total_liquidation_collateral = sum(
            (c.liquidation_value_usd for c in collaterals), Decimal(0)
        )

        return total_liquidation_collateral / total_debt

    def calculate_with_breakdown(
        self,
        position: UserPosition,
    ) -> HealthFactorBreakdown:
        """Calculate health factor with detailed breakdown and verification.

        Args:
            position: User position to analyze.

        Returns:
            HealthFactorBreakdown with full calculation details.
        """
        # Calculate totals
        total_collateral = position.total_collateral_usd
        total_liquidation_collateral = position.total_collateral_liquidation_usd
        total_debt = position.total_debt_usd

        # Calculate health factor
        calculated_hf = self.calculate_health_factor(
            position.collaterals, position.debts
        )

        # Build collateral contributions
        collateral_contributions = [
            {
                "symbol": c.symbol,
                "address": c.address,
                "amount": str(c.amount),
                "price_usd": str(c.price_usd),
                "value_usd": str(c.value_usd),
                "liquidation_threshold": str(c.liquidation_threshold),
                "liquidation_value_usd": str(c.liquidation_value_usd),
                "usage_as_collateral_enabled": c.usage_as_collateral_enabled,
                "contribution_pct": str(
                    (c.liquidation_value_usd / total_liquidation_collateral * 100)
                    if total_liquidation_collateral > 0
                    else Decimal(0)
                ),
            }
            for c in position.collaterals
        ]

        # Build debt contributions
        debt_contributions = [
            {
                "symbol": d.symbol,
                "address": d.address,
                "amount": str(d.amount),
                "price_usd": str(d.price_usd),
                "value_usd": str(d.value_usd),
                "is_stable_rate": d.is_stable_rate,
                "contribution_pct": str(
                    (d.value_usd / total_debt * 100) if total_debt > 0 else Decimal(0)
                ),
            }
            for d in position.debts
        ]

        # Verification
        on_chain_hf = position.health_factor
        verification_passed = True
        verification_diff: Decimal | None = None

        if calculated_hf is not None and on_chain_hf is not None:
            verification_diff = calculated_hf - on_chain_hf
            relative_diff = (
                abs(verification_diff / on_chain_hf) if on_chain_hf != 0 else Decimal(0)
            )
            verification_passed = relative_diff <= self.verification_tolerance

        return HealthFactorBreakdown(
            total_collateral_usd=total_collateral,
            total_collateral_liquidation_usd=total_liquidation_collateral,
            total_debt_usd=total_debt,
            calculated_health_factor=calculated_hf,
            on_chain_health_factor=on_chain_hf,
            collateral_contributions=collateral_contributions,
            debt_contributions=debt_contributions,
            verification_passed=verification_passed,
            verification_diff=verification_diff,
            verification_tolerance=self.verification_tolerance,
        )

    def simulate_price_change(
        self,
        position: UserPosition,
        asset_address: str,
        price_change_pct: Decimal,
    ) -> Decimal | None:
        """Simulate health factor after a price change.

        Args:
            position: Current position.
            asset_address: Address of asset to change price.
            price_change_pct: Price change as percentage (e.g., -10 for -10%).

        Returns:
            New health factor after price change.
        """
        # Create modified copies of assets
        modified_collaterals = []
        modified_debts = []

        price_multiplier = Decimal(1) + (price_change_pct / Decimal(100))

        for c in position.collaterals:
            if c.address.lower() == asset_address.lower():
                # Create modified collateral with new price
                modified = CollateralAsset(
                    symbol=c.symbol,
                    address=c.address,
                    decimals=c.decimals,
                    amount_raw=c.amount_raw,
                    price_usd=c.price_usd * price_multiplier,
                    asset_type=c.asset_type,
                    liquidation_threshold=c.liquidation_threshold,
                    liquidation_bonus=c.liquidation_bonus,
                    ltv=c.ltv,
                    is_active=c.is_active,
                    is_frozen=c.is_frozen,
                    usage_as_collateral_enabled=c.usage_as_collateral_enabled,
                )
                modified_collaterals.append(modified)
            else:
                modified_collaterals.append(c)

        for d in position.debts:
            if d.address.lower() == asset_address.lower():
                # Create modified debt with new price
                modified = DebtAsset(
                    symbol=d.symbol,
                    address=d.address,
                    decimals=d.decimals,
                    amount_raw=d.amount_raw,
                    price_usd=d.price_usd * price_multiplier,
                    asset_type=d.asset_type,
                    is_stable_rate=d.is_stable_rate,
                    current_rate=d.current_rate,
                )
                modified_debts.append(modified)
            else:
                modified_debts.append(d)

        return self.calculate_health_factor(modified_collaterals, modified_debts)

    def find_liquidation_price(
        self,
        position: UserPosition,
        asset_address: str,
        is_collateral: bool = True,
        precision: Decimal = Decimal("0.01"),
    ) -> Decimal | None:
        """Find price at which position becomes liquidatable.

        Args:
            position: Current position.
            asset_address: Address of asset to find threshold for.
            is_collateral: Whether asset is collateral (True) or debt (False).
            precision: Price precision as percentage.

        Returns:
            Price change percentage that triggers liquidation, or None if not applicable.
        """
        current_hf = self.calculate_health_factor(
            position.collaterals, position.debts
        )

        if current_hf is None:
            return None  # No debt

        if current_hf < Decimal(1):
            return Decimal(0)  # Already liquidatable

        # Binary search for liquidation threshold
        # For collateral: price decrease triggers liquidation
        # For debt: price increase triggers liquidation
        if is_collateral:
            low, high = Decimal(-99), Decimal(0)
        else:
            low, high = Decimal(0), Decimal(1000)

        while high - low > precision:
            mid = (low + high) / 2
            simulated_hf = self.simulate_price_change(position, asset_address, mid)

            if simulated_hf is None:
                return None

            if simulated_hf < Decimal(1):
                # Liquidatable at this price
                if is_collateral:
                    low = mid  # Need less decrease
                else:
                    high = mid  # Need less increase
            else:
                # Still healthy
                if is_collateral:
                    high = mid  # Need more decrease
                else:
                    low = mid  # Need more increase

        return (low + high) / 2

    def calculate_max_liquidation(
        self,
        position: UserPosition,
    ) -> dict[str, Any]:
        """Calculate maximum liquidation parameters.

        Aave v3 allows liquidating up to 50% of debt when HF < 1,
        or 100% when HF < 0.95 (CLOSE_FACTOR_HF_THRESHOLD).

        Args:
            position: Position to analyze.

        Returns:
            Dict with max debt to cover, collateral to receive, etc.
        """
        hf = position.health_factor or position.calculated_health_factor

        if hf is None or hf >= Decimal(1):
            return {
                "is_liquidatable": False,
                "health_factor": str(hf) if hf else "infinite",
                "max_debt_to_cover_pct": Decimal(0),
                "max_debt_to_cover_usd": Decimal(0),
            }

        # Determine close factor based on health factor
        # HF < 0.95 allows 100% liquidation
        if hf < Decimal("0.95"):
            close_factor = Decimal(1)  # 100%
        else:
            close_factor = Decimal("0.5")  # 50%

        max_debt_to_cover_usd = position.total_debt_usd * close_factor

        # Find the largest debt asset (typically what gets liquidated)
        largest_debt = max(position.debts, key=lambda d: d.value_usd, default=None)

        # Find the largest collateral (typically what's seized)
        largest_collateral = max(
            position.collaterals, key=lambda c: c.value_usd, default=None
        )

        result: dict[str, Any] = {
            "is_liquidatable": True,
            "health_factor": str(hf),
            "close_factor": str(close_factor),
            "max_debt_to_cover_pct": close_factor * 100,
            "max_debt_to_cover_usd": str(max_debt_to_cover_usd),
            "total_debt_usd": str(position.total_debt_usd),
            "total_collateral_usd": str(position.total_collateral_usd),
        }

        if largest_debt:
            result["suggested_debt_asset"] = {
                "symbol": largest_debt.symbol,
                "address": largest_debt.address,
                "value_usd": str(largest_debt.value_usd),
            }

        if largest_collateral:
            # Calculate collateral received including liquidation bonus
            collateral_to_receive = min(
                max_debt_to_cover_usd * (1 + largest_collateral.liquidation_bonus),
                largest_collateral.value_usd,
            )
            result["suggested_collateral_asset"] = {
                "symbol": largest_collateral.symbol,
                "address": largest_collateral.address,
                "value_usd": str(largest_collateral.value_usd),
                "liquidation_bonus": str(largest_collateral.liquidation_bonus),
                "collateral_to_receive_usd": str(collateral_to_receive),
            }

        return result
