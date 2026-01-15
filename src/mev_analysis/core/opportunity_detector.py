"""Opportunity detector for liquidation MEV.

Converts UserPositions into actionable LiquidationOpportunity objects by:
1. Filtering for liquidatable positions (HF < 1)
2. Calculating optimal liquidation parameters
3. Estimating gross profit, gas costs, and slippage
4. Determining if opportunity meets minimum threshold

Supports multiple detection strategies and market condition analysis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterator

from mev_analysis.core.logging import ExperimentLogger
from mev_analysis.data.health_factor import HealthFactorCalculator
from mev_analysis.data.models import (
    CollateralAsset,
    DebtAsset,
    LiquidationOpportunity,
    MarketConditions,
    UserPosition,
)


@dataclass
class DetectorConfig:
    """Configuration for opportunity detection."""

    # Minimum thresholds
    min_profit_eth: Decimal = Decimal("0.01")  # 0.01 ETH minimum
    min_debt_usd: Decimal = Decimal("100")  # Skip tiny positions
    max_debt_usd: Decimal = Decimal("1_000_000")  # Skip whale positions (risky)

    # Gas estimation
    estimated_gas_units: int = 350_000
    gas_buffer_pct: Decimal = Decimal("0.20")  # 20% buffer

    # Slippage estimation
    base_slippage_pct: Decimal = Decimal("0.1")  # 0.1% base
    slippage_per_size_factor: Decimal = Decimal("0.0001")  # per $1000

    # Close factor (Aave v3)
    default_close_factor: Decimal = Decimal("0.5")  # 50%
    full_close_factor_threshold: Decimal = Decimal("0.95")  # 100% below this HF


class OpportunityDetector:
    """Detect and analyze liquidation opportunities.

    Scans positions for liquidatable states and creates detailed
    opportunity objects with profit estimates.

    Usage:
        detector = OpportunityDetector(config, logger)
        for opportunity in detector.detect(positions, market_conditions):
            print(f"Found opportunity: {opportunity.opportunity_id}")
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            config: Detection configuration.
            logger: Optional experiment logger.
        """
        self.config = config or DetectorConfig()
        self.logger = logger
        self._hf_calculator = HealthFactorCalculator()

    def detect(
        self,
        positions: list[UserPosition],
        market_conditions: MarketConditions,
    ) -> Iterator[LiquidationOpportunity]:
        """Detect liquidation opportunities from positions.

        Args:
            positions: List of user positions to analyze.
            market_conditions: Current market conditions.

        Yields:
            LiquidationOpportunity for each actionable opportunity.
        """
        for position in positions:
            opportunity = self._analyze_position(position, market_conditions)
            if opportunity is not None:
                yield opportunity

    def detect_single(
        self,
        position: UserPosition,
        market_conditions: MarketConditions,
    ) -> LiquidationOpportunity | None:
        """Analyze a single position for liquidation opportunity.

        Args:
            position: User position to analyze.
            market_conditions: Current market conditions.

        Returns:
            LiquidationOpportunity if actionable, None otherwise.
        """
        return self._analyze_position(position, market_conditions)

    def _analyze_position(
        self,
        position: UserPosition,
        market_conditions: MarketConditions,
    ) -> LiquidationOpportunity | None:
        """Analyze position and create opportunity if liquidatable."""
        # Check if liquidatable
        if not position.is_liquidatable:
            return None

        # Get health factor
        hf = position.health_factor or position.calculated_health_factor
        if hf is None or hf >= Decimal(1):
            return None

        # Check debt size constraints
        total_debt = position.total_debt_usd
        if total_debt < self.config.min_debt_usd:
            if self.logger:
                self.logger.debug(
                    f"Skipping position {position.user_address}: debt too small",
                    {"debt_usd": str(total_debt), "min": str(self.config.min_debt_usd)},
                )
            return None

        if total_debt > self.config.max_debt_usd:
            if self.logger:
                self.logger.debug(
                    f"Skipping position {position.user_address}: debt too large",
                    {"debt_usd": str(total_debt), "max": str(self.config.max_debt_usd)},
                )
            return None

        # Find best debt/collateral pair
        debt_asset, collateral_asset = self._find_best_pair(position)
        if debt_asset is None or collateral_asset is None:
            return None

        # Calculate liquidation parameters
        close_factor = self._get_close_factor(hf)
        debt_to_cover = min(
            debt_asset.value_usd * close_factor,
            collateral_asset.value_usd / (1 + collateral_asset.liquidation_bonus),
        )

        # Calculate collateral to receive (with bonus)
        collateral_to_receive = debt_to_cover * (1 + collateral_asset.liquidation_bonus)

        # Estimate costs
        gas_cost_usd = self._estimate_gas_cost(market_conditions)
        slippage_usd = self._estimate_slippage(debt_to_cover, collateral_to_receive)

        # Calculate profit
        gross_profit_usd = collateral_to_receive - debt_to_cover
        net_profit_usd = gross_profit_usd - gas_cost_usd - slippage_usd

        # Convert to ETH
        eth_price = market_conditions.eth_price_usd
        if eth_price <= 0:
            eth_price = Decimal("2000")  # Fallback

        net_profit_eth = net_profit_usd / eth_price

        # Check if meets minimum threshold
        if net_profit_eth < self.config.min_profit_eth:
            if self.logger:
                self.logger.debug(
                    f"Opportunity below threshold: {position.user_address}",
                    {
                        "net_profit_eth": str(net_profit_eth),
                        "min_threshold": str(self.config.min_profit_eth),
                    },
                )
            return None

        # Create opportunity
        opportunity_id = f"opp_{uuid.uuid4().hex[:12]}"

        opportunity = LiquidationOpportunity(
            opportunity_id=opportunity_id,
            detected_at_block=position.block_number,
            detected_at_timestamp=datetime.now(timezone.utc),
            user_address=position.user_address,
            position_snapshot=position,
            market_conditions=market_conditions,
            debt_to_cover_usd=debt_to_cover,
            debt_asset_address=debt_asset.address,
            collateral_to_receive_usd=collateral_to_receive,
            collateral_asset_address=collateral_asset.address,
            estimated_gross_profit_usd=gross_profit_usd,
            estimated_gas_cost_usd=gas_cost_usd,
            estimated_slippage_usd=slippage_usd,
            estimated_net_profit_usd=net_profit_usd,
            estimated_net_profit_eth=net_profit_eth,
            detection_method="health_factor_scan",
            notes=f"HF={hf}, close_factor={close_factor}",
        )

        if self.logger:
            self.logger.log_event(
                "opportunity_detected",
                {
                    "opportunity_id": opportunity_id,
                    "user_address": position.user_address,
                    "health_factor": str(hf),
                    "debt_to_cover_usd": str(debt_to_cover),
                    "net_profit_eth": str(net_profit_eth),
                    "debt_asset": debt_asset.symbol,
                    "collateral_asset": collateral_asset.symbol,
                },
            )

        return opportunity

    def _find_best_pair(
        self,
        position: UserPosition,
    ) -> tuple[DebtAsset | None, CollateralAsset | None]:
        """Find the best debt/collateral pair for liquidation.

        Strategy: Maximize profit by choosing:
        - Largest debt (more to liquidate)
        - Collateral with highest bonus that covers the debt
        """
        if not position.debts or not position.collaterals:
            return None, None

        # Sort debts by value (largest first)
        sorted_debts = sorted(position.debts, key=lambda d: d.value_usd, reverse=True)

        # Sort collaterals by liquidation bonus (highest first), then by value
        usable_collaterals = [
            c for c in position.collaterals
            if c.usage_as_collateral_enabled and c.value_usd > 0
        ]
        sorted_collaterals = sorted(
            usable_collaterals,
            key=lambda c: (c.liquidation_bonus, c.value_usd),
            reverse=True,
        )

        if not sorted_debts or not sorted_collaterals:
            return None, None

        # Return best pair
        return sorted_debts[0], sorted_collaterals[0]

    def _get_close_factor(self, health_factor: Decimal) -> Decimal:
        """Determine close factor based on health factor.

        Aave v3: 50% normally, 100% when HF < 0.95
        """
        if health_factor < self.config.full_close_factor_threshold:
            return Decimal(1)  # 100%
        return self.config.default_close_factor  # 50%

    def _estimate_gas_cost(self, market_conditions: MarketConditions) -> Decimal:
        """Estimate gas cost in USD."""
        gas_price_gwei = market_conditions.gas_price_gwei
        gas_units = self.config.estimated_gas_units

        # Gas cost in ETH
        gas_cost_eth = Decimal(gas_units) * gas_price_gwei / Decimal("1e9")

        # Add buffer
        gas_cost_eth *= (1 + self.config.gas_buffer_pct)

        # Convert to USD
        eth_price = market_conditions.eth_price_usd
        if eth_price <= 0:
            eth_price = Decimal("2000")

        return gas_cost_eth * eth_price

    def _estimate_slippage(
        self,
        debt_to_cover: Decimal,
        collateral_to_receive: Decimal,
    ) -> Decimal:
        """Estimate slippage based on position size."""
        # Base slippage
        slippage_pct = self.config.base_slippage_pct

        # Add size-based slippage
        size_factor = float(debt_to_cover) / 1000  # per $1000
        slippage_pct += self.config.slippage_per_size_factor * Decimal(str(size_factor))

        # Apply to collateral (what we're receiving)
        return collateral_to_receive * slippage_pct / 100


def filter_actionable_opportunities(
    opportunities: list[LiquidationOpportunity],
    min_profit_eth: Decimal = Decimal("0.01"),
    min_capture_prob: Decimal | None = None,
) -> list[LiquidationOpportunity]:
    """Filter opportunities to only actionable ones.

    Args:
        opportunities: List of opportunities to filter.
        min_profit_eth: Minimum profit threshold.
        min_capture_prob: Optional minimum capture probability.

    Returns:
        Filtered list of actionable opportunities.
    """
    filtered = []
    for opp in opportunities:
        if opp.estimated_net_profit_eth >= min_profit_eth:
            filtered.append(opp)
    return filtered
