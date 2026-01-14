"""Aave v3 contract interface for Arbitrum.

Provides typed access to Aave v3 protocol data:
- User account data (health factor, collateral, debt)
- Reserve configuration
- Oracle prices

All interactions respect SAFE_MODE constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from web3 import Web3
from web3.contract import Contract
from web3.types import BlockIdentifier

from mev_analysis.core.safe_mode import SafeMode
from mev_analysis.data.constants import (
    AAVE_BASE_CURRENCY_UNIT,
    AAVE_V3_ARBITRUM_ADDRESSES,
    AAVE_V3_ARBITRUM_SEPOLIA_ADDRESSES,
    ARBITRUM_MAINNET_CHAIN_ID,
    ARBITRUM_SEPOLIA_CHAIN_ID,
    ERC20_ABI,
    HEALTH_FACTOR_DECIMALS,
    ORACLE_ABI,
    POOL_ABI,
    POOL_DATA_PROVIDER_ABI,
)
from mev_analysis.data.models import (
    AssetType,
    CollateralAsset,
    DebtAsset,
    MarketConditions,
    UserPosition,
)


@dataclass
class UserAccountData:
    """Raw user account data from Aave Pool contract."""

    total_collateral_base: int  # In base currency (USD with 8 decimals)
    total_debt_base: int
    available_borrows_base: int
    current_liquidation_threshold: int  # In basis points (e.g., 8250 = 82.5%)
    ltv: int  # In basis points
    health_factor: int  # 18 decimals, 10^18 = 1.0

    @property
    def total_collateral_usd(self) -> Decimal:
        """Total collateral in USD."""
        return Decimal(self.total_collateral_base) / Decimal(AAVE_BASE_CURRENCY_UNIT)

    @property
    def total_debt_usd(self) -> Decimal:
        """Total debt in USD."""
        return Decimal(self.total_debt_base) / Decimal(AAVE_BASE_CURRENCY_UNIT)

    @property
    def health_factor_decimal(self) -> Decimal:
        """Health factor as decimal (1.0 = healthy threshold)."""
        return Decimal(self.health_factor) / Decimal(10**HEALTH_FACTOR_DECIMALS)

    @property
    def is_liquidatable(self) -> bool:
        """Check if position is liquidatable (HF < 1)."""
        return self.health_factor < 10**HEALTH_FACTOR_DECIMALS


@dataclass
class ReserveConfig:
    """Reserve configuration data."""

    asset_address: str
    symbol: str
    decimals: int
    ltv: int  # Basis points
    liquidation_threshold: int  # Basis points
    liquidation_bonus: int  # Basis points (e.g., 10500 = 5% bonus)
    reserve_factor: int
    usage_as_collateral_enabled: bool
    borrowing_enabled: bool
    stable_borrow_rate_enabled: bool
    is_active: bool
    is_frozen: bool


class AaveV3Client:
    """Client for interacting with Aave v3 protocol.

    Requires SAFE_MODE to be initialized and environment verified.

    Usage:
        client = AaveV3Client(web3_instance)
        account_data = client.get_user_account_data(user_address)
        position = client.get_user_position(user_address, block_number)
    """

    def __init__(self, web3: Web3) -> None:
        """Initialize Aave v3 client.

        Args:
            web3: Connected Web3 instance.

        Raises:
            SafeModeError: If SAFE_MODE not initialized or environment not verified.
        """
        self.web3 = web3
        self._safe_mode = SafeMode.get_instance()
        self._safe_mode.require_verification()

        # Determine addresses based on chain ID
        chain_id = self.web3.eth.chain_id
        if chain_id == ARBITRUM_MAINNET_CHAIN_ID:
            self._addresses = AAVE_V3_ARBITRUM_ADDRESSES
        elif chain_id == ARBITRUM_SEPOLIA_CHAIN_ID:
            self._addresses = AAVE_V3_ARBITRUM_SEPOLIA_ADDRESSES
        else:
            # For local forks, assume mainnet addresses
            self._addresses = AAVE_V3_ARBITRUM_ADDRESSES

        # Initialize contracts
        self._pool: Contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(self._addresses["pool"]),
            abi=POOL_ABI,
        )
        self._data_provider: Contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(self._addresses["pool_data_provider"]),
            abi=POOL_DATA_PROVIDER_ABI,
        )
        self._oracle: Contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(self._addresses["oracle"]),
            abi=ORACLE_ABI,
        )

        # Cache for reserve configs
        self._reserve_configs: dict[str, ReserveConfig] = {}
        self._reserves_list: list[str] | None = None

    def get_user_account_data(
        self,
        user_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> UserAccountData:
        """Get user account data from Pool contract.

        Args:
            user_address: User wallet address.
            block_identifier: Block number or 'latest'.

        Returns:
            UserAccountData with collateral, debt, and health factor.
        """
        checksum_address = Web3.to_checksum_address(user_address)
        result = self._pool.functions.getUserAccountData(checksum_address).call(
            block_identifier=block_identifier
        )

        return UserAccountData(
            total_collateral_base=result[0],
            total_debt_base=result[1],
            available_borrows_base=result[2],
            current_liquidation_threshold=result[3],
            ltv=result[4],
            health_factor=result[5],
        )

    def get_reserves_list(
        self, block_identifier: BlockIdentifier = "latest"
    ) -> list[str]:
        """Get list of all reserve asset addresses.

        Args:
            block_identifier: Block number or 'latest'.

        Returns:
            List of reserve token addresses.
        """
        if self._reserves_list is None:
            self._reserves_list = self._pool.functions.getReservesList().call(
                block_identifier=block_identifier
            )
        return self._reserves_list

    def get_reserve_config(
        self,
        asset_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> ReserveConfig:
        """Get reserve configuration for an asset.

        Args:
            asset_address: Token address.
            block_identifier: Block number or 'latest'.

        Returns:
            ReserveConfig with LTV, liquidation parameters, etc.
        """
        cache_key = f"{asset_address}_{block_identifier}"
        if cache_key in self._reserve_configs:
            return self._reserve_configs[cache_key]

        checksum_address = Web3.to_checksum_address(asset_address)

        # Get configuration data
        config = self._data_provider.functions.getReserveConfigurationData(
            checksum_address
        ).call(block_identifier=block_identifier)

        # Get symbol
        token_contract = self.web3.eth.contract(
            address=checksum_address, abi=ERC20_ABI
        )
        try:
            symbol = token_contract.functions.symbol().call(
                block_identifier=block_identifier
            )
        except Exception:
            symbol = "UNKNOWN"

        reserve_config = ReserveConfig(
            asset_address=asset_address.lower(),
            symbol=symbol,
            decimals=config[0],
            ltv=config[1],
            liquidation_threshold=config[2],
            liquidation_bonus=config[3],
            reserve_factor=config[4],
            usage_as_collateral_enabled=config[5],
            borrowing_enabled=config[6],
            stable_borrow_rate_enabled=config[7],
            is_active=config[8],
            is_frozen=config[9],
        )

        self._reserve_configs[cache_key] = reserve_config
        return reserve_config

    def get_asset_price(
        self,
        asset_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> Decimal:
        """Get asset price in USD from Aave Oracle.

        Args:
            asset_address: Token address.
            block_identifier: Block number or 'latest'.

        Returns:
            Price in USD (8 decimals converted to Decimal).
        """
        checksum_address = Web3.to_checksum_address(asset_address)
        price_raw = self._oracle.functions.getAssetPrice(checksum_address).call(
            block_identifier=block_identifier
        )
        return Decimal(price_raw) / Decimal(AAVE_BASE_CURRENCY_UNIT)

    def get_user_reserve_data(
        self,
        asset_address: str,
        user_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> dict[str, Any]:
        """Get user's data for a specific reserve.

        Args:
            asset_address: Token address.
            user_address: User wallet address.
            block_identifier: Block number or 'latest'.

        Returns:
            Dict with aToken balance, debt amounts, and collateral status.
        """
        result = self._data_provider.functions.getUserReserveData(
            Web3.to_checksum_address(asset_address),
            Web3.to_checksum_address(user_address),
        ).call(block_identifier=block_identifier)

        return {
            "atoken_balance": result[0],
            "stable_debt": result[1],
            "variable_debt": result[2],
            "principal_stable_debt": result[3],
            "scaled_variable_debt": result[4],
            "stable_borrow_rate": result[5],
            "liquidity_rate": result[6],
            "stable_rate_last_updated": result[7],
            "usage_as_collateral_enabled": result[8],
        }

    def get_user_emode(
        self,
        user_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> int:
        """Get user's E-mode category.

        Args:
            user_address: User wallet address.
            block_identifier: Block number or 'latest'.

        Returns:
            E-mode category ID (0 = no E-mode).
        """
        checksum_address = Web3.to_checksum_address(user_address)
        return self._pool.functions.getUserEMode(checksum_address).call(
            block_identifier=block_identifier
        )

    def get_user_position(
        self,
        user_address: str,
        block_identifier: BlockIdentifier = "latest",
    ) -> UserPosition:
        """Get complete user position with all collaterals and debts.

        Args:
            user_address: User wallet address.
            block_identifier: Block number or 'latest'.

        Returns:
            UserPosition with all assets and calculated metrics.
        """
        # Resolve block number
        if block_identifier == "latest":
            block = self.web3.eth.get_block("latest")
            block_number = block["number"]
            timestamp = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)
        else:
            block_number = int(block_identifier)  # type: ignore[arg-type]
            block = self.web3.eth.get_block(block_number)
            timestamp = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)

        # Get on-chain account data for health factor
        account_data = self.get_user_account_data(user_address, block_identifier)

        # Get user's E-mode
        e_mode = self.get_user_emode(user_address, block_identifier)

        # Get all reserves and user data for each
        reserves = self.get_reserves_list(block_identifier)

        collaterals: list[CollateralAsset] = []
        debts: list[DebtAsset] = []

        for reserve_address in reserves:
            reserve_config = self.get_reserve_config(reserve_address, block_identifier)
            user_reserve = self.get_user_reserve_data(
                reserve_address, user_address, block_identifier
            )

            # Skip if user has no position in this reserve
            has_collateral = user_reserve["atoken_balance"] > 0
            has_debt = (
                user_reserve["stable_debt"] > 0 or user_reserve["variable_debt"] > 0
            )

            if not has_collateral and not has_debt:
                continue

            # Get price
            price_usd = self.get_asset_price(reserve_address, block_identifier)

            # Determine asset type (simplified heuristic)
            asset_type = self._classify_asset(reserve_config.symbol)

            # Add collateral if present
            if has_collateral:
                collateral = CollateralAsset(
                    symbol=reserve_config.symbol,
                    address=reserve_address.lower(),
                    decimals=reserve_config.decimals,
                    amount_raw=user_reserve["atoken_balance"],
                    price_usd=price_usd,
                    asset_type=asset_type,
                    liquidation_threshold=Decimal(reserve_config.liquidation_threshold)
                    / Decimal(10000),
                    liquidation_bonus=Decimal(reserve_config.liquidation_bonus - 10000)
                    / Decimal(10000),  # Convert from 10500 to 0.05
                    ltv=Decimal(reserve_config.ltv) / Decimal(10000),
                    is_active=reserve_config.is_active,
                    is_frozen=reserve_config.is_frozen,
                    usage_as_collateral_enabled=user_reserve[
                        "usage_as_collateral_enabled"
                    ],
                )
                collaterals.append(collateral)

            # Add debt if present
            if has_debt:
                total_debt = (
                    user_reserve["stable_debt"] + user_reserve["variable_debt"]
                )
                is_stable = user_reserve["stable_debt"] > user_reserve["variable_debt"]

                debt = DebtAsset(
                    symbol=reserve_config.symbol,
                    address=reserve_address.lower(),
                    decimals=reserve_config.decimals,
                    amount_raw=total_debt,
                    price_usd=price_usd,
                    asset_type=asset_type,
                    is_stable_rate=is_stable,
                    current_rate=Decimal(user_reserve["stable_borrow_rate"])
                    / Decimal(10**27)
                    if is_stable
                    else Decimal(0),  # Simplified - would need variable rate
                )
                debts.append(debt)

        return UserPosition(
            user_address=user_address.lower(),
            block_number=block_number,
            timestamp=timestamp,
            collaterals=collaterals,
            debts=debts,
            health_factor=account_data.health_factor_decimal,
            health_factor_source="on_chain",
            e_mode_category=e_mode,
        )

    def _classify_asset(self, symbol: str) -> AssetType:
        """Classify asset type based on symbol.

        Args:
            symbol: Token symbol.

        Returns:
            AssetType classification.
        """
        symbol_upper = symbol.upper()

        # Stablecoins
        if symbol_upper in ["USDC", "USDC.E", "USDT", "DAI", "FRAX", "LUSD", "GHO"]:
            return AssetType.STABLE

        # ETH correlated
        if symbol_upper in ["WETH", "ETH", "STETH", "WSTETH", "RETH", "CBETH"]:
            return AssetType.ETH_CORRELATED

        # BTC correlated
        if symbol_upper in ["WBTC", "BTC", "TBTC"]:
            return AssetType.BTC_CORRELATED

        return AssetType.VOLATILE

    def get_market_conditions(
        self,
        block_identifier: BlockIdentifier = "latest",
    ) -> MarketConditions:
        """Get current market conditions.

        Args:
            block_identifier: Block number or 'latest'.

        Returns:
            MarketConditions snapshot.
        """
        # Get block data
        if block_identifier == "latest":
            block = self.web3.eth.get_block("latest")
        else:
            block = self.web3.eth.get_block(int(block_identifier))  # type: ignore[arg-type]

        block_number = block["number"]
        timestamp = datetime.fromtimestamp(block["timestamp"], tz=timezone.utc)

        # Gas prices
        base_fee = block.get("baseFeePerGas", 0)
        gas_price = self.web3.eth.gas_price

        # ETH price (WETH on Arbitrum)
        try:
            weth_address = "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
            eth_price = self.get_asset_price(weth_address, block_identifier)
        except Exception:
            eth_price = Decimal(0)

        # Block utilization
        gas_used = block.get("gasUsed", 0)
        gas_limit = block.get("gasLimit", 1)
        utilization = Decimal(gas_used) / Decimal(gas_limit) if gas_limit > 0 else Decimal(0)

        return MarketConditions(
            block_number=block_number,
            timestamp=timestamp,
            gas_price_gwei=Decimal(gas_price) / Decimal(10**9),
            base_fee_gwei=Decimal(base_fee) / Decimal(10**9) if base_fee else None,
            priority_fee_gwei=None,  # Would need to estimate
            block_utilization=utilization,
            pending_tx_count=None,
            eth_price_usd=eth_price,
            eth_price_change_1h=None,
            eth_price_change_24h=None,
            total_liquidations_24h=None,
            avg_liquidation_size_usd=None,
        )
