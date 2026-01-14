"""Tests for bot archetypes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from mev_analysis.simulation.bots import (
    Bot,
    BotConfig,
    BotType,
    BackrunnerBot,
    FrontrunnerBot,
    GasSensitiveBot,
    RandomBot,
    SlippageAwareBot,
    TailEventBot,
    create_bot,
    create_default_bot_pool,
)


class TestBotConfig:
    """Tests for BotConfig."""

    def test_default_config(self) -> None:
        """Should create config with defaults."""
        config = BotConfig(bot_type=BotType.FRONTRUNNER, bot_id="test_bot")
        assert config.activation_probability == 0.5
        assert config.execution_latency_ms == 100.0
        assert config.max_position_usd == 100_000.0

    def test_custom_config(self) -> None:
        """Should accept custom parameters."""
        config = BotConfig(
            bot_type=BotType.GAS_SENSITIVE,
            bot_id="custom_bot",
            activation_probability=0.8,
            max_gas_gwei=0.3,
        )
        assert config.activation_probability == 0.8
        assert config.max_gas_gwei == 0.3


class TestFrontrunnerBot:
    """Tests for FrontrunnerBot."""

    @pytest.fixture
    def bot(self) -> FrontrunnerBot:
        """Create frontrunner bot with fixed seed."""
        config = BotConfig(
            bot_type=BotType.FRONTRUNNER,
            bot_id="test_frontrunner",
            activation_probability=0.7,
            execution_latency_ms=50,
        )
        return FrontrunnerBot(config, seed=42)

    def test_activation_for_profitable_opportunity(self, bot: FrontrunnerBot) -> None:
        """Should activate for profitable opportunities."""
        # Run multiple times to account for stochasticity
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("100"),
                gas_price_gwei=Decimal("0.1"),
                current_liquidity_usd=Decimal("10000"),
            ):
                activations += 1

        # Should activate frequently (70%+ given config)
        assert activations >= 50

    def test_no_activation_for_unprofitable(self, bot: FrontrunnerBot) -> None:
        """Should not activate for unprofitable opportunities."""
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("-10"),
                gas_price_gwei=Decimal("0.1"),
                current_liquidity_usd=Decimal("10000"),
            ):
                activations += 1

        assert activations == 0

    def test_execution_time_frontrunning(self, bot: FrontrunnerBot) -> None:
        """Should execute before base time (frontrunning)."""
        base_time = 100.0
        times = [bot.calculate_execution_time(base_time) for _ in range(100)]

        # Most executions should be before base time
        before_count = sum(1 for t in times if t < base_time)
        assert before_count >= 80  # At least 80% should frontrun

    def test_max_position_sizing(self, bot: FrontrunnerBot) -> None:
        """Should take maximum position within capital."""
        position = bot.calculate_position_size(
            max_debt_usd=Decimal("50000"),
            available_capital_usd=Decimal("100000"),
        )
        assert position == Decimal("50000")  # Takes full opportunity


class TestBackrunnerBot:
    """Tests for BackrunnerBot."""

    @pytest.fixture
    def bot(self) -> BackrunnerBot:
        """Create backrunner bot."""
        config = BotConfig(
            bot_type=BotType.BACKRUNNER,
            bot_id="test_backrunner",
            execution_latency_ms=150,
        )
        return BackrunnerBot(config, seed=42)

    def test_execution_time_after_base(self, bot: BackrunnerBot) -> None:
        """Should execute after base time."""
        base_time = 100.0
        times = [bot.calculate_execution_time(base_time) for _ in range(100)]

        # All executions should be after base time
        after_count = sum(1 for t in times if t > base_time)
        assert after_count == 100

    def test_partial_position_sizing(self, bot: BackrunnerBot) -> None:
        """Should take partial positions."""
        positions = []
        for i in range(50):
            bot.reset_rng(42 + i)
            pos = bot.calculate_position_size(
                max_debt_usd=Decimal("100000"),
                available_capital_usd=Decimal("100000"),
            )
            positions.append(float(pos))

        # Should take 50-100% of max
        assert all(50000 <= p <= 100000 for p in positions)


class TestGasSensitiveBot:
    """Tests for GasSensitiveBot."""

    @pytest.fixture
    def bot(self) -> GasSensitiveBot:
        """Create gas-sensitive bot."""
        config = BotConfig(
            bot_type=BotType.GAS_SENSITIVE,
            bot_id="test_gas_sensitive",
            activation_probability=0.8,
            max_gas_gwei=0.5,
        )
        return GasSensitiveBot(config, seed=42)

    def test_no_activation_high_gas(self, bot: GasSensitiveBot) -> None:
        """Should not activate when gas is high."""
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("100"),
                gas_price_gwei=Decimal("1.0"),  # Above threshold
                current_liquidity_usd=Decimal("10000"),
            ):
                activations += 1

        assert activations == 0

    def test_activation_low_gas(self, bot: GasSensitiveBot) -> None:
        """Should activate when gas is low."""
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("100"),
                gas_price_gwei=Decimal("0.2"),  # Below threshold
                current_liquidity_usd=Decimal("10000"),
            ):
                activations += 1

        assert activations >= 70  # High activation rate expected


class TestSlippageAwareBot:
    """Tests for SlippageAwareBot."""

    @pytest.fixture
    def bot(self) -> SlippageAwareBot:
        """Create slippage-aware bot."""
        config = BotConfig(
            bot_type=BotType.SLIPPAGE_AWARE,
            bot_id="test_slippage_aware",
            activation_probability=0.6,
            max_slippage_pct=1.5,
        )
        return SlippageAwareBot(config, seed=42)

    def test_no_activation_high_slippage(self, bot: SlippageAwareBot) -> None:
        """Should not activate when slippage would be high."""
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("10000"),  # Large relative to liquidity
                gas_price_gwei=Decimal("0.1"),
                current_liquidity_usd=Decimal("1000"),  # Small liquidity
            ):
                activations += 1

        # High slippage scenario - should rarely activate
        assert activations < 20

    def test_activation_low_slippage(self, bot: SlippageAwareBot) -> None:
        """Should activate when slippage is acceptable."""
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("100"),  # Small relative to liquidity
                gas_price_gwei=Decimal("0.1"),
                current_liquidity_usd=Decimal("100000"),  # Large liquidity
            ):
                activations += 1

        assert activations >= 50


class TestTailEventBot:
    """Tests for TailEventBot."""

    @pytest.fixture
    def bot(self) -> TailEventBot:
        """Create tail event bot."""
        config = BotConfig(
            bot_type=BotType.TAIL_EVENT,
            bot_id="test_tail_event",
            tail_event_probability=0.05,  # 5% for testing
            tail_event_multiplier=5.0,
            max_position_usd=500_000,
        )
        return TailEventBot(config, seed=42)

    def test_rare_activation(self, bot: TailEventBot) -> None:
        """Should rarely activate (tail event)."""
        activations = 0
        for i in range(1000):  # Need more iterations for rare events
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("100"),
                gas_price_gwei=Decimal("0.1"),
                current_liquidity_usd=Decimal("10000"),
            ):
                activations += 1

        # Should be around 5% of 1000 = 50, with some variance
        assert 20 <= activations <= 100

    def test_large_position_when_activated(self, bot: TailEventBot) -> None:
        """Should take large positions when activated."""
        position = bot.calculate_position_size(
            max_debt_usd=Decimal("10000"),
            available_capital_usd=Decimal("500000"),
        )

        # Should multiply by tail event factor
        assert position >= Decimal("50000")  # 10000 * 5


class TestRandomBot:
    """Tests for RandomBot."""

    @pytest.fixture
    def bot(self) -> RandomBot:
        """Create random bot."""
        config = BotConfig(
            bot_type=BotType.RANDOM,
            bot_id="test_random",
            activation_probability=0.3,
        )
        return RandomBot(config, seed=42)

    def test_random_activation(self, bot: RandomBot) -> None:
        """Should activate randomly regardless of opportunity quality."""
        # Even for negative EV, should still activate sometimes
        activations = 0
        for i in range(100):
            bot.reset_rng(42 + i)
            if bot.should_activate(
                opportunity_ev_usd=Decimal("-100"),  # Negative EV
                gas_price_gwei=Decimal("10.0"),  # High gas
                current_liquidity_usd=Decimal("100"),  # Low liquidity
            ):
                activations += 1

        # Should still activate ~30% of time
        assert 15 <= activations <= 50

    def test_variable_execution_timing(self, bot: RandomBot) -> None:
        """Should have highly variable execution timing."""
        base_time = 100.0
        times = [bot.calculate_execution_time(base_time) for _ in range(100)]

        # Should have both before and after base time
        before_count = sum(1 for t in times if t < base_time)
        assert 20 <= before_count <= 80  # Roughly split


class TestBotFactory:
    """Tests for bot factory functions."""

    def test_create_bot(self) -> None:
        """Should create correct bot type."""
        bot = create_bot(BotType.FRONTRUNNER, "test_bot", seed=42)
        assert isinstance(bot, FrontrunnerBot)
        assert bot.bot_id == "test_bot"
        assert bot.bot_type == BotType.FRONTRUNNER

    def test_create_default_bot_pool(self) -> None:
        """Should create pool of 10+ diverse bots."""
        bots = create_default_bot_pool(base_seed=42)

        assert len(bots) >= 10

        # Check diversity of bot types
        types = {bot.bot_type for bot in bots}
        assert BotType.FRONTRUNNER in types
        assert BotType.BACKRUNNER in types
        assert BotType.RANDOM in types
        assert BotType.GAS_SENSITIVE in types
        assert BotType.SLIPPAGE_AWARE in types
        assert BotType.TAIL_EVENT in types

    def test_bot_pool_unique_ids(self) -> None:
        """Should have unique bot IDs."""
        bots = create_default_bot_pool()
        ids = [bot.bot_id for bot in bots]
        assert len(ids) == len(set(ids))  # All unique


class TestBotAction:
    """Tests for BotAction execution."""

    def test_execute_action(self) -> None:
        """Should produce valid action on execute."""
        bot = create_bot(
            BotType.FRONTRUNNER,
            "test_bot",
            seed=42,
            activation_probability=1.0,  # Always activate
        )

        action = bot.execute(
            opportunity_id="opp_1",
            opportunity_ev_usd=Decimal("100"),
            max_debt_usd=Decimal("1000"),
            gas_price_gwei=Decimal("0.1"),
            current_liquidity_usd=Decimal("10000"),
            base_time_ms=100.0,
            liquidation_bonus_pct=Decimal("0.05"),
        )

        assert action.bot_id == "test_bot"
        assert action.opportunity_id == "opp_1"
        assert action.action_type == "execute"
        assert action.success
        assert action.debt_covered_usd > 0
        assert action.collateral_seized_usd > action.debt_covered_usd  # Due to bonus

    def test_skip_action_no_activation(self) -> None:
        """Should skip when not activating."""
        bot = create_bot(
            BotType.GAS_SENSITIVE,
            "test_bot",
            seed=42,
            max_gas_gwei=0.05,  # Very low threshold
        )

        action = bot.execute(
            opportunity_id="opp_1",
            opportunity_ev_usd=Decimal("100"),
            max_debt_usd=Decimal("1000"),
            gas_price_gwei=Decimal("1.0"),  # High gas
            current_liquidity_usd=Decimal("10000"),
            base_time_ms=100.0,
            liquidation_bonus_pct=Decimal("0.05"),
        )

        assert action.action_type == "skip"
        assert not action.success
        assert action.failure_reason == "did_not_activate"


class TestDeterministicReplay:
    """Tests for deterministic replay with seeds."""

    def test_same_seed_same_results(self) -> None:
        """Same seed should produce identical results."""
        bot1 = create_bot(BotType.RANDOM, "bot1", seed=12345)
        bot2 = create_bot(BotType.RANDOM, "bot2", seed=12345)

        results1 = []
        results2 = []

        for _ in range(10):
            results1.append(bot1.calculate_execution_time(100.0))
            results2.append(bot2.calculate_execution_time(100.0))

        assert results1 == results2

    def test_different_seeds_different_results(self) -> None:
        """Different seeds should produce different results."""
        bot1 = create_bot(BotType.RANDOM, "bot1", seed=12345)
        bot2 = create_bot(BotType.RANDOM, "bot2", seed=54321)

        results1 = [bot1.calculate_execution_time(100.0) for _ in range(10)]
        results2 = [bot2.calculate_execution_time(100.0) for _ in range(10)]

        assert results1 != results2

    def test_reset_rng_reproducibility(self) -> None:
        """Resetting RNG should reproduce results."""
        bot = create_bot(BotType.RANDOM, "bot", seed=42)

        # First run
        results1 = [bot.calculate_execution_time(100.0) for _ in range(5)]

        # Reset and run again
        bot.reset_rng(42)
        results2 = [bot.calculate_execution_time(100.0) for _ in range(5)]

        assert results1 == results2
