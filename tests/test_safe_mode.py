"""Tests for SAFE_MODE enforcement."""

from __future__ import annotations

import pytest

from mev_analysis.core.safe_mode import (
    ARBITRUM_MAINNET_CHAIN_ID,
    SAFE_CHAIN_IDS,
    SafeMode,
    SafeModeError,
)


class TestSafeModeInitialization:
    """Tests for SafeMode initialization."""

    def test_initialize_with_safe_mode_enabled(self, reset_safe_mode: None) -> None:
        """SafeMode should initialize when SAFE_MODE=true."""
        safe_mode = SafeMode.initialize()
        assert safe_mode is not None
        assert safe_mode.config.enabled is True

    def test_initialize_fails_without_safe_mode_env(
        self, reset_safe_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SafeMode should fail to initialize when SAFE_MODE is not set."""
        monkeypatch.delenv("SAFE_MODE", raising=False)

        with pytest.raises(SafeModeError, match="SAFE_MODE must be set to 'true'"):
            SafeMode.initialize()

    def test_initialize_fails_with_safe_mode_false(
        self, reset_safe_mode: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SafeMode should fail to initialize when SAFE_MODE=false."""
        monkeypatch.setenv("SAFE_MODE", "false")

        with pytest.raises(SafeModeError, match="SAFE_MODE must be set to 'true'"):
            SafeMode.initialize()

    def test_singleton_pattern(self, reset_safe_mode: None) -> None:
        """SafeMode should be a singleton."""
        instance1 = SafeMode.initialize()
        instance2 = SafeMode.initialize()
        assert instance1 is instance2

    def test_get_instance_before_initialize(self, reset_safe_mode: None) -> None:
        """get_instance should fail before initialization."""
        with pytest.raises(SafeModeError, match="SAFE_MODE not initialized"):
            SafeMode.get_instance()


class TestEnvironmentVerification:
    """Tests for environment verification."""

    def test_verify_testnet_chain_id(self, safe_mode_instance: SafeMode) -> None:
        """Should accept known testnet chain IDs."""
        # Arbitrum Sepolia
        assert safe_mode_instance.verify_environment(421614) is True
        assert safe_mode_instance._verified_chain_id == 421614

    def test_verify_local_chain_id(
        self, reset_safe_mode: None, safe_mode_instance: SafeMode
    ) -> None:
        """Should accept local development chain IDs."""
        # Hardhat/Anvil
        assert safe_mode_instance.verify_environment(31337) is True

    def test_verify_mainnet_without_fork_fails(self, safe_mode_instance: SafeMode) -> None:
        """Should reject mainnet chain ID without fork flag."""
        with pytest.raises(SafeModeError, match="is_fork=False"):
            safe_mode_instance.verify_environment(ARBITRUM_MAINNET_CHAIN_ID, is_fork=False)

    def test_verify_mainnet_fork_requires_block_number(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should require block number for mainnet fork verification."""
        with pytest.raises(SafeModeError, match="block_number"):
            safe_mode_instance.verify_environment(
                ARBITRUM_MAINNET_CHAIN_ID, is_fork=True, block_number=None
            )

    def test_verify_mainnet_fork_with_block_number(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should accept mainnet fork with block number."""
        assert safe_mode_instance.verify_environment(
            ARBITRUM_MAINNET_CHAIN_ID, is_fork=True, block_number=12345678
        ) is True
        assert safe_mode_instance._is_fork is True

    def test_verify_unknown_chain_id_fails(self, safe_mode_instance: SafeMode) -> None:
        """Should reject unknown chain IDs."""
        with pytest.raises(SafeModeError, match="Unknown chain ID"):
            safe_mode_instance.verify_environment(999999)

    def test_verification_hash_generated(self, safe_mode_instance: SafeMode) -> None:
        """Should generate verification hash after verification."""
        safe_mode_instance.verify_environment(421614)
        assert safe_mode_instance.verification_hash is not None
        assert len(safe_mode_instance.verification_hash) == 16


class TestEVCapVerification:
    """Tests for EV cap enforcement."""

    def test_ev_within_cap(self, safe_mode_instance: SafeMode) -> None:
        """Should accept EV within cap."""
        assert safe_mode_instance.verify_ev_cap(0.5) is True

    def test_ev_exceeds_cap(self, safe_mode_instance: SafeMode) -> None:
        """Should reject EV exceeding cap."""
        with pytest.raises(SafeModeError, match="exceeds safety cap"):
            safe_mode_instance.verify_ev_cap(10.0)

    def test_ev_exactly_at_cap(self, safe_mode_instance: SafeMode) -> None:
        """Should accept EV exactly at cap."""
        assert safe_mode_instance.verify_ev_cap(1.0) is True


class TestPrivateKeySafety:
    """Tests for private key safety checks."""

    def test_no_private_key_is_safe(self, safe_mode_instance: SafeMode) -> None:
        """Should accept no private key."""
        assert safe_mode_instance.verify_private_key_safety(None) is True

    def test_private_key_requires_environment_verification(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should require environment verification before using private key."""
        with pytest.raises(SafeModeError, match="verify_environment"):
            safe_mode_instance.verify_private_key_safety("0x1234")

    def test_private_key_allowed_on_testnet(self, safe_mode_instance: SafeMode) -> None:
        """Should allow private key on testnet."""
        safe_mode_instance.verify_environment(421614)
        assert safe_mode_instance.verify_private_key_safety("0x1234") is True


class TestVerificationContext:
    """Tests for verification context."""

    def test_verification_context_before_verification(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should return context even before verification."""
        ctx = safe_mode_instance.get_verification_context()
        assert ctx["safe_mode_enabled"] is True
        assert ctx["verified_chain_id"] is None

    def test_verification_context_after_verification(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should return full context after verification."""
        safe_mode_instance.verify_environment(421614)
        ctx = safe_mode_instance.get_verification_context()
        assert ctx["verified_chain_id"] == 421614
        assert ctx["verification_hash"] is not None

    def test_require_verification_fails_before(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should fail if verification not done."""
        with pytest.raises(SafeModeError, match="requires verified environment"):
            safe_mode_instance.require_verification()

    def test_require_verification_passes_after(
        self, safe_mode_instance: SafeMode
    ) -> None:
        """Should pass after verification."""
        safe_mode_instance.verify_environment(421614)
        safe_mode_instance.require_verification()  # Should not raise
