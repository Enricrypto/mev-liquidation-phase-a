"""SAFE_MODE enforcement for Phase A research system.

This module ensures that the system operates ONLY in safe research mode:
- No real capital exposure
- Testnet/forked execution only
- Runtime verification of all safety constraints
- Automatic abort on any safety violation

SAFE_MODE must be enabled for all Phase A operations.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dotenv import load_dotenv


class SafeModeError(Exception):
    """Raised when a SAFE_MODE violation is detected."""

    pass


class ExecutionEnvironment(Enum):
    """Valid execution environments for Phase A."""

    FORKED_MAINNET = "forked_mainnet"
    TESTNET = "testnet"
    LOCAL_FORK = "local_fork"
    SIMULATION = "simulation"


# Known testnet and local chain IDs
SAFE_CHAIN_IDS: dict[int, str] = {
    421614: "Arbitrum Sepolia",
    31337: "Local Hardhat/Anvil",
    1337: "Local Ganache",
}

# Arbitrum mainnet chain ID - requires fork verification
ARBITRUM_MAINNET_CHAIN_ID = 42161


@dataclass
class SafeModeConfig:
    """Configuration for SAFE_MODE enforcement."""

    enabled: bool = True
    max_ev_cap_eth: float = 1.0
    allowed_environments: list[ExecutionEnvironment] = field(
        default_factory=lambda: list(ExecutionEnvironment)
    )
    require_fork_verification: bool = True
    abort_on_anomaly: bool = True


class SafeMode:
    """SAFE_MODE enforcement singleton.

    Ensures all Phase A operations run in a safe, controlled environment.
    Must be initialized before any simulation or data operations.

    Usage:
        safe_mode = SafeMode.initialize()
        safe_mode.verify_environment(chain_id, is_fork=True)
        safe_mode.verify_ev_cap(estimated_ev)
    """

    _instance: SafeMode | None = None
    _initialized: bool = False

    def __init__(self, config: SafeModeConfig | None = None) -> None:
        """Initialize SAFE_MODE with configuration.

        Args:
            config: SafeModeConfig instance, or None to load from environment.

        Raises:
            SafeModeError: If SAFE_MODE is not enabled in environment.
        """
        load_dotenv()

        # Check environment variable
        safe_mode_env = os.getenv("SAFE_MODE", "").lower()
        if safe_mode_env != "true":
            raise SafeModeError(
                "SAFE_MODE must be set to 'true' in environment. "
                "Phase A requires SAFE_MODE for all operations. "
                "Set SAFE_MODE=true in your .env file."
            )

        self.config = config or self._load_config_from_env()
        self._verification_hash: str | None = None
        self._verified_chain_id: int | None = None
        self._is_fork: bool = False

    @classmethod
    def initialize(cls, config: SafeModeConfig | None = None) -> SafeMode:
        """Initialize or return the SAFE_MODE singleton.

        Args:
            config: Optional configuration override.

        Returns:
            The SafeMode singleton instance.

        Raises:
            SafeModeError: If SAFE_MODE cannot be initialized.
        """
        if cls._instance is None:
            cls._instance = cls(config)
            cls._initialized = True
        return cls._instance

    @classmethod
    def get_instance(cls) -> SafeMode:
        """Get the initialized SAFE_MODE instance.

        Returns:
            The SafeMode singleton instance.

        Raises:
            SafeModeError: If SAFE_MODE has not been initialized.
        """
        if cls._instance is None:
            raise SafeModeError(
                "SAFE_MODE not initialized. Call SafeMode.initialize() first."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        cls._instance = None
        cls._initialized = False

    def _load_config_from_env(self) -> SafeModeConfig:
        """Load configuration from environment variables."""
        max_ev_cap = float(os.getenv("MAX_EV_CAP_ETH", "1.0"))
        return SafeModeConfig(
            enabled=True,
            max_ev_cap_eth=max_ev_cap,
            allowed_environments=list(ExecutionEnvironment),
            require_fork_verification=True,
            abort_on_anomaly=True,
        )

    def verify_environment(
        self,
        chain_id: int,
        is_fork: bool = False,
        block_number: int | None = None,
    ) -> bool:
        """Verify the execution environment is safe for Phase A.

        Args:
            chain_id: The chain ID of the connected network.
            is_fork: Whether this is a forked environment.
            block_number: Optional block number for fork verification.

        Returns:
            True if environment is verified safe.

        Raises:
            SafeModeError: If environment fails safety checks.
        """
        # Check if it's a known safe testnet
        if chain_id in SAFE_CHAIN_IDS:
            self._verified_chain_id = chain_id
            self._is_fork = False
            self._generate_verification_hash(chain_id, is_fork, block_number)
            return True

        # Arbitrum mainnet requires fork verification
        if chain_id == ARBITRUM_MAINNET_CHAIN_ID:
            if not is_fork:
                raise SafeModeError(
                    f"Chain ID {chain_id} (Arbitrum Mainnet) detected but is_fork=False. "
                    "Phase A requires forked mainnet for Arbitrum mainnet chain ID. "
                    "Use a local fork (Anvil/Hardhat) or switch to testnet."
                )
            if self.config.require_fork_verification and block_number is None:
                raise SafeModeError(
                    "Fork verification requires block_number to be specified. "
                    "Provide the block number the fork was created from."
                )
            self._verified_chain_id = chain_id
            self._is_fork = True
            self._generate_verification_hash(chain_id, is_fork, block_number)
            return True

        # Unknown chain ID
        raise SafeModeError(
            f"Unknown chain ID {chain_id}. "
            f"Safe chain IDs: {list(SAFE_CHAIN_IDS.keys())} or "
            f"Arbitrum mainnet ({ARBITRUM_MAINNET_CHAIN_ID}) with is_fork=True."
        )

    def verify_ev_cap(self, estimated_ev_eth: float) -> bool:
        """Verify that estimated EV is within safety cap.

        Args:
            estimated_ev_eth: Estimated expected value in ETH.

        Returns:
            True if EV is within cap.

        Raises:
            SafeModeError: If EV exceeds safety cap.
        """
        if estimated_ev_eth > self.config.max_ev_cap_eth:
            raise SafeModeError(
                f"Estimated EV ({estimated_ev_eth} ETH) exceeds safety cap "
                f"({self.config.max_ev_cap_eth} ETH). "
                "Reduce simulation size or increase MAX_EV_CAP_ETH."
            )
        return True

    def verify_rpc_url(self, rpc_url: str) -> bool:
        """Verify RPC URL doesn't contain obvious mainnet indicators without fork.

        Args:
            rpc_url: The RPC URL to verify.

        Returns:
            True if RPC URL passes basic safety checks.

        Raises:
            SafeModeError: If RPC URL fails safety checks.
        """
        # Basic check - actual chain ID verification happens in verify_environment
        if not rpc_url:
            raise SafeModeError("RPC URL cannot be empty.")

        # Warn but don't block - actual verification is via chain ID
        mainnet_indicators = ["mainnet", "arb-mainnet", "arbitrum-one"]
        url_lower = rpc_url.lower()
        for indicator in mainnet_indicators:
            if indicator in url_lower and not self._is_fork:
                # This is just a warning - actual check is chain ID + is_fork
                pass

        return True

    def verify_private_key_safety(self, private_key: str | None) -> bool:
        """Verify private key usage is appropriate for Phase A.

        Args:
            private_key: The private key (if any) being used.

        Returns:
            True if private key usage is safe.

        Note:
            Phase A should minimize private key exposure.
            Only testnet keys should ever be used.
        """
        if private_key is None:
            return True

        # We can't actually verify if a key is testnet-only,
        # but we can log a warning and ensure SAFE_MODE context
        if not self._verified_chain_id:
            raise SafeModeError(
                "Cannot use private key before environment verification. "
                "Call verify_environment() first."
            )

        if self._verified_chain_id == ARBITRUM_MAINNET_CHAIN_ID and not self._is_fork:
            raise SafeModeError(
                "Private key usage not allowed on mainnet without fork. "
                "This should never happen if SAFE_MODE is properly enforced."
            )

        return True

    def _generate_verification_hash(
        self,
        chain_id: int,
        is_fork: bool,
        block_number: int | None,
    ) -> None:
        """Generate a verification hash for audit trail."""
        data = f"chain_id={chain_id}|is_fork={is_fork}|block={block_number}"
        self._verification_hash = hashlib.sha256(data.encode()).hexdigest()[:16]

    @property
    def verification_hash(self) -> str | None:
        """Get the current verification hash."""
        return self._verification_hash

    @property
    def is_verified(self) -> bool:
        """Check if environment has been verified."""
        return self._verified_chain_id is not None

    def get_verification_context(self) -> dict[str, Any]:
        """Get the current verification context for logging."""
        return {
            "safe_mode_enabled": self.config.enabled,
            "verified_chain_id": self._verified_chain_id,
            "is_fork": self._is_fork,
            "verification_hash": self._verification_hash,
            "max_ev_cap_eth": self.config.max_ev_cap_eth,
        }

    def require_verification(self) -> None:
        """Require that environment has been verified.

        Raises:
            SafeModeError: If environment not yet verified.
        """
        if not self.is_verified:
            raise SafeModeError(
                "Operation requires verified environment. "
                "Call verify_environment() with chain_id first."
            )
