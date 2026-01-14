"""Bot archetypes for MEV simulation.

Implements stochastic and adversarial bot behaviors for stress-testing
liquidation capture probability. Each archetype models a different
competitive strategy observed in MEV markets.

Bot Archetypes:
- Frontrunner: Competes to execute before our simulation
- Backrunner: Executes after, consuming liquidity
- Random/Noise: Stochastic behavior representing minor bots
- GasSensitive: Only acts when gas is favorable
- SlippageAware: Considers liquidity depth before acting
- TailEvent: Rare but high-impact actions (black swan)

All bots are parameterized for deterministic replay via random seeds.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class BotType(str, Enum):
    """Bot archetype classification."""

    FRONTRUNNER = "frontrunner"
    BACKRUNNER = "backrunner"
    RANDOM = "random"
    GAS_SENSITIVE = "gas_sensitive"
    SLIPPAGE_AWARE = "slippage_aware"
    TAIL_EVENT = "tail_event"


@dataclass
class BotConfig:
    """Configuration parameters for a bot instance.

    All parameters support deterministic replay when combined with a seed.
    """

    bot_type: BotType
    bot_id: str

    # Activation probability per simulation step (0-1)
    activation_probability: float = 0.5

    # Execution latency in simulated milliseconds
    execution_latency_ms: float = 100.0
    execution_latency_std: float = 50.0  # Standard deviation for randomness

    # Gas sensitivity (for gas-sensitive bots)
    max_gas_gwei: float = 1.0  # Only act if gas below this

    # Slippage sensitivity (for slippage-aware bots)
    max_slippage_pct: float = 2.0  # Only act if slippage below this

    # Capital constraints
    max_position_usd: float = 100_000.0  # Maximum position size

    # Tail event parameters
    tail_event_probability: float = 0.01  # Rare event trigger
    tail_event_multiplier: float = 10.0  # Impact multiplier when triggered

    # Aggressiveness (affects gas bidding, position sizing)
    aggressiveness: float = 0.5  # 0-1 scale


@dataclass
class BotAction:
    """Represents a bot's action in a simulation step."""

    bot_id: str
    bot_type: BotType
    action_type: str  # "execute", "skip", "partial"
    timestamp_ms: float  # Simulated execution time

    # Impact on opportunity
    opportunity_id: str | None = None
    debt_covered_usd: Decimal = Decimal(0)
    collateral_seized_usd: Decimal = Decimal(0)

    # Costs incurred
    gas_used: int = 0
    gas_price_gwei: Decimal = Decimal(0)
    slippage_pct: Decimal = Decimal(0)

    # Outcome
    success: bool = False
    failure_reason: str | None = None

    # Metadata for logging
    metadata: dict[str, Any] = field(default_factory=dict)


class Bot(ABC):
    """Abstract base class for bot archetypes.

    Each bot implementation defines:
    - Whether to activate on a given opportunity
    - Execution timing and priority
    - Position sizing and risk parameters
    - Impact on the simulation state
    """

    def __init__(self, config: BotConfig, seed: int | None = None) -> None:
        """Initialize bot with configuration.

        Args:
            config: Bot configuration parameters.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def bot_id(self) -> str:
        """Unique identifier for this bot."""
        return self.config.bot_id

    @property
    def bot_type(self) -> BotType:
        """Bot archetype type."""
        return self.config.bot_type

    def reset_rng(self, seed: int | None = None) -> None:
        """Reset random number generator for replay.

        Args:
            seed: New seed, or use original if None.
        """
        self._rng = random.Random(seed if seed is not None else self.seed)

    @abstractmethod
    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Determine if bot should activate for this opportunity.

        Args:
            opportunity_ev_usd: Estimated value of the opportunity.
            gas_price_gwei: Current gas price.
            current_liquidity_usd: Available liquidity.

        Returns:
            True if bot will attempt to capture the opportunity.
        """
        pass

    @abstractmethod
    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Calculate when this bot would execute.

        Args:
            base_time_ms: Base execution time for the opportunity.

        Returns:
            Bot's execution timestamp in milliseconds.
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Calculate position size for this opportunity.

        Args:
            max_debt_usd: Maximum debt that can be covered.
            available_capital_usd: Bot's available capital.

        Returns:
            Amount of debt to cover in USD.
        """
        pass

    def execute(
        self,
        opportunity_id: str,
        opportunity_ev_usd: Decimal,
        max_debt_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
        base_time_ms: float,
        liquidation_bonus_pct: Decimal,
    ) -> BotAction:
        """Execute bot's action for an opportunity.

        Args:
            opportunity_id: Unique ID of the opportunity.
            opportunity_ev_usd: Estimated value.
            max_debt_usd: Maximum debt to cover.
            gas_price_gwei: Current gas price.
            current_liquidity_usd: Available liquidity.
            base_time_ms: Base execution timestamp.
            liquidation_bonus_pct: Liquidation bonus percentage.

        Returns:
            BotAction describing the outcome.
        """
        # Check activation
        if not self.should_activate(opportunity_ev_usd, gas_price_gwei, current_liquidity_usd):
            return BotAction(
                bot_id=self.bot_id,
                bot_type=self.bot_type,
                action_type="skip",
                timestamp_ms=base_time_ms,
                opportunity_id=opportunity_id,
                success=False,
                failure_reason="did_not_activate",
            )

        # Calculate execution timing
        exec_time = self.calculate_execution_time(base_time_ms)

        # Calculate position size
        position_size = self.calculate_position_size(
            max_debt_usd, Decimal(self.config.max_position_usd)
        )

        if position_size <= 0:
            return BotAction(
                bot_id=self.bot_id,
                bot_type=self.bot_type,
                action_type="skip",
                timestamp_ms=exec_time,
                opportunity_id=opportunity_id,
                success=False,
                failure_reason="insufficient_capital",
            )

        # Calculate collateral received
        collateral_received = position_size * (1 + liquidation_bonus_pct)

        # Estimate gas used (simplified)
        gas_used = 300_000 + self._rng.randint(0, 100_000)

        # Calculate slippage based on position size vs liquidity
        slippage_pct = self._calculate_slippage(position_size, current_liquidity_usd)

        return BotAction(
            bot_id=self.bot_id,
            bot_type=self.bot_type,
            action_type="execute",
            timestamp_ms=exec_time,
            opportunity_id=opportunity_id,
            debt_covered_usd=position_size,
            collateral_seized_usd=collateral_received,
            gas_used=gas_used,
            gas_price_gwei=gas_price_gwei,
            slippage_pct=slippage_pct,
            success=True,
            metadata={
                "activation_roll": self._rng.random(),
                "latency_adjustment": exec_time - base_time_ms,
            },
        )

    def _calculate_slippage(
        self, position_size: Decimal, liquidity: Decimal
    ) -> Decimal:
        """Estimate slippage based on position size relative to liquidity."""
        if liquidity <= 0:
            return Decimal("10.0")  # High slippage if no liquidity

        # Simple linear model: slippage increases with position/liquidity ratio
        ratio = float(position_size / liquidity)
        base_slippage = ratio * 5.0  # 5% slippage at 100% of liquidity
        noise = self._rng.gauss(0, 0.1)
        return Decimal(str(max(0, base_slippage + noise)))


class FrontrunnerBot(Bot):
    """Frontrunner bot: competes to execute before others.

    Characteristics:
    - Low latency execution
    - High activation probability for profitable opportunities
    - Aggressive gas bidding
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Activate if opportunity looks profitable."""
        # Always try for profitable opportunities
        if opportunity_ev_usd <= 0:
            return False

        # Probabilistic activation based on config
        roll = self._rng.random()
        # Higher activation for larger opportunities
        ev_bonus = min(float(opportunity_ev_usd) / 1000, 0.3)
        return roll < (self.config.activation_probability + ev_bonus)

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Execute as fast as possible (before base time)."""
        # Frontrunners try to execute before the opportunity
        latency = self._rng.gauss(
            self.config.execution_latency_ms * 0.5,  # Faster than average
            self.config.execution_latency_std * 0.5,
        )
        # Can execute before base_time (frontrunning)
        return base_time_ms - abs(latency)

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Take maximum position within capital constraints."""
        # Aggressive: take as much as possible
        return min(max_debt_usd, available_capital_usd)


class BackrunnerBot(Bot):
    """Backrunner bot: executes after others, consuming remaining value.

    Characteristics:
    - Slightly delayed execution
    - Focuses on leftover opportunities
    - May increase slippage for others
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Activate for opportunities with remaining value."""
        if opportunity_ev_usd <= 0:
            return False

        roll = self._rng.random()
        return roll < self.config.activation_probability

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Execute shortly after the base time."""
        # Backrunners wait a bit
        latency = self._rng.gauss(
            self.config.execution_latency_ms * 1.5,
            self.config.execution_latency_std,
        )
        return base_time_ms + abs(latency)

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Take remaining position."""
        # Conservative: take partial position
        target = min(max_debt_usd, available_capital_usd)
        return target * Decimal(str(0.5 + self._rng.random() * 0.5))


class RandomBot(Bot):
    """Random/Noise bot: stochastic behavior simulating minor market participants.

    Characteristics:
    - Unpredictable activation
    - Variable timing
    - Prevents overfitting to deterministic assumptions
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Random activation regardless of opportunity quality."""
        roll = self._rng.random()
        return roll < self.config.activation_probability

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Random execution timing."""
        # Highly variable timing
        latency = self._rng.gauss(
            self.config.execution_latency_ms,
            self.config.execution_latency_std * 2,  # High variance
        )
        offset = self._rng.choice([-1, 1]) * abs(latency)
        return base_time_ms + offset

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Random position sizing."""
        max_position = min(max_debt_usd, available_capital_usd)
        return max_position * Decimal(str(self._rng.random()))


class GasSensitiveBot(Bot):
    """Gas-sensitive bot: only acts when gas is favorable.

    Characteristics:
    - Strict gas price threshold
    - Higher activity during low-gas periods
    - Models economically rational actors
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Activate only if gas is below threshold."""
        if float(gas_price_gwei) > self.config.max_gas_gwei:
            return False

        if opportunity_ev_usd <= 0:
            return False

        roll = self._rng.random()
        # Higher activation when gas is very low
        gas_bonus = max(0, (self.config.max_gas_gwei - float(gas_price_gwei)) / self.config.max_gas_gwei) * 0.3
        return roll < (self.config.activation_probability + gas_bonus)

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Standard execution timing."""
        latency = self._rng.gauss(
            self.config.execution_latency_ms,
            self.config.execution_latency_std,
        )
        return base_time_ms + latency

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Size based on gas efficiency."""
        max_position = min(max_debt_usd, available_capital_usd)
        # More conservative position sizing
        return max_position * Decimal(str(0.3 + self._rng.random() * 0.5))


class SlippageAwareBot(Bot):
    """Slippage-aware bot: considers liquidity depth before acting.

    Characteristics:
    - Avoids high-slippage situations
    - Sizes positions relative to liquidity
    - Models sophisticated MEV actors
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Activate if slippage would be acceptable."""
        if opportunity_ev_usd <= 0:
            return False

        # Estimate slippage based on opportunity size vs liquidity
        estimated_slippage = float(opportunity_ev_usd / current_liquidity_usd) * 100 if current_liquidity_usd > 0 else 100

        if estimated_slippage > self.config.max_slippage_pct:
            return False

        roll = self._rng.random()
        # Bonus for low-slippage opportunities
        slippage_bonus = max(0, (self.config.max_slippage_pct - estimated_slippage) / self.config.max_slippage_pct) * 0.2
        return roll < (self.config.activation_probability + slippage_bonus)

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Moderate execution timing."""
        latency = self._rng.gauss(
            self.config.execution_latency_ms,
            self.config.execution_latency_std,
        )
        return base_time_ms + latency

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Size position to minimize slippage impact."""
        max_position = min(max_debt_usd, available_capital_usd)
        # Conservative: limit position to avoid excessive slippage
        return max_position * Decimal(str(0.2 + self._rng.random() * 0.4))


class TailEventBot(Bot):
    """Tail-event bot: rare but high-impact actions.

    Characteristics:
    - Very low activation probability
    - Massive position sizes when triggered
    - Models black swan scenarios and whale activity
    """

    def should_activate(
        self,
        opportunity_ev_usd: Decimal,
        gas_price_gwei: Decimal,
        current_liquidity_usd: Decimal,
    ) -> bool:
        """Rarely activate, but when triggered, always act."""
        roll = self._rng.random()
        # Very low probability, but ignores other factors when triggered
        return roll < self.config.tail_event_probability

    def calculate_execution_time(self, base_time_ms: float) -> float:
        """Variable timing - can be very fast or delayed."""
        # Bimodal distribution: either very fast or delayed
        if self._rng.random() < 0.5:
            # Fast execution (whale with infrastructure)
            latency = self._rng.gauss(
                self.config.execution_latency_ms * 0.3,
                self.config.execution_latency_std * 0.3,
            )
            return base_time_ms - abs(latency)
        else:
            # Delayed (reactive whale)
            latency = self._rng.gauss(
                self.config.execution_latency_ms * 2,
                self.config.execution_latency_std,
            )
            return base_time_ms + abs(latency)

    def calculate_position_size(
        self,
        max_debt_usd: Decimal,
        available_capital_usd: Decimal,
    ) -> Decimal:
        """Take massive positions when activated."""
        max_position = min(max_debt_usd, available_capital_usd)
        # Multiply by tail event factor
        multiplier = Decimal(str(self.config.tail_event_multiplier))
        target = max_position * multiplier
        # Cap at available capital (but this is a tail event bot, so capital is high)
        return min(target, available_capital_usd * Decimal("5"))


def create_bot(bot_type: BotType, bot_id: str, seed: int | None = None, **kwargs: Any) -> Bot:
    """Factory function to create bot instances.

    Args:
        bot_type: Type of bot to create.
        bot_id: Unique identifier.
        seed: Random seed for reproducibility.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured Bot instance.
    """
    config = BotConfig(bot_type=bot_type, bot_id=bot_id, **kwargs)

    bot_classes: dict[BotType, type[Bot]] = {
        BotType.FRONTRUNNER: FrontrunnerBot,
        BotType.BACKRUNNER: BackrunnerBot,
        BotType.RANDOM: RandomBot,
        BotType.GAS_SENSITIVE: GasSensitiveBot,
        BotType.SLIPPAGE_AWARE: SlippageAwareBot,
        BotType.TAIL_EVENT: TailEventBot,
    }

    bot_class = bot_classes[bot_type]
    return bot_class(config, seed)


def create_default_bot_pool(base_seed: int = 42) -> list[Bot]:
    """Create the default pool of 10+ bots for simulation.

    Returns a diverse set of bot archetypes as specified in the design doc.

    Args:
        base_seed: Base seed for reproducibility.

    Returns:
        List of configured Bot instances.
    """
    bots: list[Bot] = []

    # 2 Frontrunners (aggressive competition)
    for i in range(2):
        bots.append(create_bot(
            BotType.FRONTRUNNER,
            f"frontrunner_{i}",
            seed=base_seed + i,
            activation_probability=0.7,
            execution_latency_ms=50,
            aggressiveness=0.8,
        ))

    # 2 Backrunners
    for i in range(2):
        bots.append(create_bot(
            BotType.BACKRUNNER,
            f"backrunner_{i}",
            seed=base_seed + 10 + i,
            activation_probability=0.5,
            execution_latency_ms=150,
        ))

    # 2 Random/Noise bots
    for i in range(2):
        bots.append(create_bot(
            BotType.RANDOM,
            f"random_{i}",
            seed=base_seed + 20 + i,
            activation_probability=0.3,
            execution_latency_ms=100,
        ))

    # 2 Gas-sensitive bots
    for i in range(2):
        bots.append(create_bot(
            BotType.GAS_SENSITIVE,
            f"gas_sensitive_{i}",
            seed=base_seed + 30 + i,
            activation_probability=0.6,
            max_gas_gwei=0.5,
        ))

    # 1 Slippage-aware bot
    bots.append(create_bot(
        BotType.SLIPPAGE_AWARE,
        "slippage_aware_0",
        seed=base_seed + 40,
        activation_probability=0.5,
        max_slippage_pct=1.5,
    ))

    # 1 Tail-event bot
    bots.append(create_bot(
        BotType.TAIL_EVENT,
        "tail_event_0",
        seed=base_seed + 50,
        tail_event_probability=0.02,
        tail_event_multiplier=5.0,
        max_position_usd=500_000,
    ))

    return bots
