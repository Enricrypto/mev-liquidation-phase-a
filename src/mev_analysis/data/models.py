"""Data models for MEV liquidation analysis.

Pydantic models for representing:
- User positions on Aave v3
- Market state and conditions
- Liquidation opportunities
- Simulation results

All models support serialization for hash-chained logging and replay.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AssetType(str, Enum):
    """Classification of asset types."""

    STABLE = "stable"
    VOLATILE = "volatile"
    ETH_CORRELATED = "eth_correlated"
    BTC_CORRELATED = "btc_correlated"


class PositionStatus(str, Enum):
    """Status of a user position."""

    HEALTHY = "healthy"
    AT_RISK = "at_risk"  # Health factor < 1.5
    LIQUIDATABLE = "liquidatable"  # Health factor < 1.0
    LIQUIDATED = "liquidated"


class Asset(BaseModel):
    """Represents a single asset in a position."""

    symbol: str
    address: str = Field(description="Token contract address")
    decimals: int = Field(ge=0, le=18)
    amount_raw: int = Field(description="Raw amount in token decimals")
    price_usd: Decimal = Field(ge=0, description="Price in USD")
    asset_type: AssetType = AssetType.VOLATILE

    @property
    def amount(self) -> Decimal:
        """Human-readable amount."""
        return Decimal(self.amount_raw) / Decimal(10**self.decimals)

    @property
    def value_usd(self) -> Decimal:
        """Value in USD."""
        return self.amount * self.price_usd

    def model_dump_with_computed(self) -> dict[str, Any]:
        """Dump model including computed properties."""
        data = self.model_dump()
        data["amount"] = str(self.amount)
        data["value_usd"] = str(self.value_usd)
        return data


class CollateralAsset(Asset):
    """Collateral asset with Aave-specific parameters."""

    liquidation_threshold: Decimal = Field(
        ge=0, le=1, description="Liquidation threshold (e.g., 0.825 = 82.5%)"
    )
    liquidation_bonus: Decimal = Field(
        ge=0, description="Liquidation bonus (e.g., 0.05 = 5%)"
    )
    ltv: Decimal = Field(ge=0, le=1, description="Loan-to-value ratio")
    is_active: bool = True
    is_frozen: bool = False
    usage_as_collateral_enabled: bool = True

    @property
    def liquidation_value_usd(self) -> Decimal:
        """Value adjusted by liquidation threshold."""
        if not self.usage_as_collateral_enabled:
            return Decimal(0)
        return self.value_usd * self.liquidation_threshold


class DebtAsset(Asset):
    """Debt asset (borrowed)."""

    is_stable_rate: bool = False
    current_rate: Decimal = Field(ge=0, description="Current borrow rate (APY)")


class UserPosition(BaseModel):
    """Complete user position on Aave v3.

    Represents a snapshot of a user's collateral and debt at a specific block.
    """

    user_address: str = Field(description="User wallet address")
    block_number: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    collaterals: list[CollateralAsset] = Field(default_factory=list)
    debts: list[DebtAsset] = Field(default_factory=list)

    # Cached calculations (optional, can be recomputed)
    health_factor: Decimal | None = Field(
        default=None, description="Health factor from on-chain or calculated"
    )
    health_factor_source: str = Field(
        default="unknown", description="Source: 'on_chain', 'calculated', 'unknown'"
    )

    # E-mode support
    e_mode_category: int = Field(default=0, description="E-mode category ID (0 = none)")

    @field_validator("user_address", mode="before")
    @classmethod
    def normalize_address(cls, v: str) -> str:
        """Normalize address to checksummed format."""
        if not v.startswith("0x"):
            v = "0x" + v
        return v.lower()  # Store lowercase, checksum on output if needed

    @property
    def total_collateral_usd(self) -> Decimal:
        """Total collateral value in USD."""
        return sum((c.value_usd for c in self.collaterals), Decimal(0))

    @property
    def total_collateral_liquidation_usd(self) -> Decimal:
        """Total collateral adjusted by liquidation thresholds."""
        return sum((c.liquidation_value_usd for c in self.collaterals), Decimal(0))

    @property
    def total_debt_usd(self) -> Decimal:
        """Total debt value in USD."""
        return sum((d.value_usd for d in self.debts), Decimal(0))

    @property
    def calculated_health_factor(self) -> Decimal | None:
        """Calculate health factor from position data.

        Health Factor = Sum(Collateral_i * LiquidationThreshold_i) / TotalDebt

        Returns None if no debt (infinite health factor).
        """
        if self.total_debt_usd == 0:
            return None  # No debt = infinite health factor
        return self.total_collateral_liquidation_usd / self.total_debt_usd

    @property
    def status(self) -> PositionStatus:
        """Determine position status based on health factor."""
        hf = self.health_factor or self.calculated_health_factor
        if hf is None:
            return PositionStatus.HEALTHY
        if hf < Decimal("1.0"):
            return PositionStatus.LIQUIDATABLE
        if hf < Decimal("1.5"):
            return PositionStatus.AT_RISK
        return PositionStatus.HEALTHY

    @property
    def is_liquidatable(self) -> bool:
        """Check if position can be liquidated."""
        return self.status == PositionStatus.LIQUIDATABLE


class MarketConditions(BaseModel):
    """Market conditions at a specific point in time.

    Used for stratified analysis and regression modeling.
    """

    block_number: int
    timestamp: datetime

    # Gas conditions
    gas_price_gwei: Decimal = Field(ge=0)
    base_fee_gwei: Decimal | None = None
    priority_fee_gwei: Decimal | None = None

    # Network conditions
    block_utilization: Decimal = Field(
        ge=0, le=1, default=Decimal("0.5"), description="Block gas usage ratio"
    )
    pending_tx_count: int | None = None

    # Market volatility indicators
    eth_price_usd: Decimal = Field(ge=0)
    eth_price_change_1h: Decimal | None = Field(
        default=None, description="ETH price change in last hour (%)"
    )
    eth_price_change_24h: Decimal | None = Field(
        default=None, description="ETH price change in last 24h (%)"
    )

    # Aave-specific
    total_liquidations_24h: int | None = None
    avg_liquidation_size_usd: Decimal | None = None

    # Time-based features (for time-of-day analysis)
    @property
    def hour_of_day(self) -> int:
        """Hour of day in UTC."""
        return self.timestamp.hour

    @property
    def day_of_week(self) -> int:
        """Day of week (0=Monday)."""
        return self.timestamp.weekday()

    @property
    def is_weekend(self) -> bool:
        """Check if weekend."""
        return self.day_of_week >= 5


class LiquidationOpportunity(BaseModel):
    """A detected liquidation opportunity.

    Combines position data with market conditions and potential profit analysis.
    """

    opportunity_id: str = Field(description="Unique identifier for this opportunity")
    detected_at_block: int
    detected_at_timestamp: datetime

    # Position reference
    user_address: str
    position_snapshot: UserPosition

    # Market context
    market_conditions: MarketConditions

    # Liquidation parameters
    debt_to_cover_usd: Decimal = Field(
        ge=0, description="Amount of debt to cover in USD"
    )
    debt_asset_address: str
    collateral_to_receive_usd: Decimal = Field(
        ge=0, description="Collateral to receive in USD (including bonus)"
    )
    collateral_asset_address: str

    # Profit estimation (pre-simulation)
    estimated_gross_profit_usd: Decimal = Field(description="Gross profit before costs")
    estimated_gas_cost_usd: Decimal = Field(ge=0)
    estimated_slippage_usd: Decimal = Field(ge=0, default=Decimal(0))
    estimated_net_profit_usd: Decimal = Field(description="Net profit after costs")

    # Conversion to ETH for threshold comparison
    estimated_net_profit_eth: Decimal = Field(
        description="Net profit in ETH for threshold comparison"
    )

    # Metadata
    detection_method: str = Field(
        default="health_factor_scan", description="How this opportunity was detected"
    )
    notes: str = Field(default="")

    @property
    def is_actionable(self) -> bool:
        """Check if opportunity meets minimum threshold (0.01 ETH)."""
        return self.estimated_net_profit_eth >= Decimal("0.01")

    @property
    def profit_margin_pct(self) -> Decimal:
        """Profit margin as percentage of debt covered."""
        if self.debt_to_cover_usd == 0:
            return Decimal(0)
        return (self.estimated_net_profit_usd / self.debt_to_cover_usd) * 100


class SimulationResult(BaseModel):
    """Result of a liquidation simulation.

    Captures EV, variance, and capture probability from Monte Carlo simulation.
    """

    opportunity_id: str
    simulation_id: str

    # Simulation parameters
    num_iterations: int = Field(ge=1)
    random_seed: int
    num_competing_bots: int = Field(ge=0)

    # EV results
    mean_ev_eth: Decimal
    std_ev_eth: Decimal
    min_ev_eth: Decimal
    max_ev_eth: Decimal

    # Confidence intervals (bootstrap)
    ev_ci_lower_95: Decimal
    ev_ci_upper_95: Decimal

    # Capture probability
    capture_probability: Decimal = Field(
        ge=0, le=1, description="Probability of successful capture"
    )
    capture_ci_lower_95: Decimal = Field(ge=0, le=1)
    capture_ci_upper_95: Decimal = Field(ge=0, le=1)

    # Cost breakdown
    mean_gas_cost_eth: Decimal
    mean_slippage_eth: Decimal

    # Failure analysis
    failure_count: int = Field(ge=0)
    failure_rate: Decimal = Field(ge=0, le=1)
    failure_reasons: dict[str, int] = Field(
        default_factory=dict, description="Failure reason -> count"
    )

    # Timing
    simulation_duration_ms: int = Field(ge=0)

    @property
    def is_profitable(self) -> bool:
        """Check if mean EV is positive with 95% confidence."""
        return self.ev_ci_lower_95 > Decimal(0)

    @property
    def meets_capture_threshold(self) -> bool:
        """Check if capture probability meets 3% threshold."""
        return self.capture_probability >= Decimal("0.03")


class KnownPosition(BaseModel):
    """Input format for known positions (CSV/JSON import).

    Simplified format for loading historical positions.
    """

    user_address: str
    block_number: int | None = None  # If None, fetch at latest/specified block
    label: str = Field(default="", description="Optional label for this position")
    source: str = Field(default="manual", description="Where this address came from")

    @field_validator("user_address", mode="before")
    @classmethod
    def normalize_address(cls, v: str) -> str:
        """Normalize address."""
        if not v.startswith("0x"):
            v = "0x" + v
        return v.lower()
