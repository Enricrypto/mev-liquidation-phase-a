"""Constants for Aave v3 on Arbitrum.

Contract addresses, ABIs, and protocol parameters.
"""

from __future__ import annotations

# =============================================================================
# Arbitrum Mainnet Addresses (Aave v3)
# =============================================================================

AAVE_V3_ARBITRUM_ADDRESSES = {
    # Core contracts
    "pool": "0x794a61358D6845594F94dc1DB02A252b5b4814aD",
    "pool_data_provider": "0x69FA688f1Dc47d4B5d8029D5a35FB7a548310654",
    "pool_addresses_provider": "0xa97684ead0e402dC232d5A977953DF7ECBaB3CDb",
    "oracle": "0xb56c2F0B653B2e0b10C9b928C8580Ac5Df02C7C7",
    # Tokens
    "aave_token": "0xba5DdD1f9d7F570dc94a51479a000E3BCE967196",
}

# =============================================================================
# Arbitrum Sepolia Testnet Addresses (Aave v3)
# =============================================================================

AAVE_V3_ARBITRUM_SEPOLIA_ADDRESSES = {
    "pool": "0xBfC91D59fdAA134A4ED45f7B584cAf96D7792Eff",
    "pool_data_provider": "0x4bE03883ef24B0AEb79C6b9e24b48E0dF7B41b48",
    "pool_addresses_provider": "0x036D8E89A6cB64b44Cc4a9f5a65E1A6f2A0296F8",
    "oracle": "0x2A6F14C1c8b0E3C097f8cd1Cb81f5D0D4Da7d3d7",
}

# =============================================================================
# Chain IDs
# =============================================================================

ARBITRUM_MAINNET_CHAIN_ID = 42161
ARBITRUM_SEPOLIA_CHAIN_ID = 421614

# =============================================================================
# Minimal ABIs (only functions we need)
# =============================================================================

# Pool contract - getUserAccountData
POOL_ABI = [
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "getUserAccountData",
        "outputs": [
            {"name": "totalCollateralBase", "type": "uint256"},
            {"name": "totalDebtBase", "type": "uint256"},
            {"name": "availableBorrowsBase", "type": "uint256"},
            {"name": "currentLiquidationThreshold", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "healthFactor", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getReservesList",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "asset", "type": "address"}],
        "name": "getReserveData",
        "outputs": [
            {
                "components": [
                    {"name": "configuration", "type": "uint256"},
                    {"name": "liquidityIndex", "type": "uint128"},
                    {"name": "currentLiquidityRate", "type": "uint128"},
                    {"name": "variableBorrowIndex", "type": "uint128"},
                    {"name": "currentVariableBorrowRate", "type": "uint128"},
                    {"name": "currentStableBorrowRate", "type": "uint128"},
                    {"name": "lastUpdateTimestamp", "type": "uint40"},
                    {"name": "id", "type": "uint16"},
                    {"name": "aTokenAddress", "type": "address"},
                    {"name": "stableDebtTokenAddress", "type": "address"},
                    {"name": "variableDebtTokenAddress", "type": "address"},
                    {"name": "interestRateStrategyAddress", "type": "address"},
                    {"name": "accruedToTreasury", "type": "uint128"},
                    {"name": "unbacked", "type": "uint128"},
                    {"name": "isolationModeTotalDebt", "type": "uint128"},
                ],
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "id", "type": "uint8"}],
        "name": "getEModeCategoryData",
        "outputs": [
            {
                "components": [
                    {"name": "ltv", "type": "uint16"},
                    {"name": "liquidationThreshold", "type": "uint16"},
                    {"name": "liquidationBonus", "type": "uint16"},
                    {"name": "priceSource", "type": "address"},
                    {"name": "label", "type": "string"},
                ],
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "user", "type": "address"}],
        "name": "getUserEMode",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# PoolDataProvider - detailed user reserve data
POOL_DATA_PROVIDER_ABI = [
    {
        "inputs": [
            {"name": "asset", "type": "address"},
            {"name": "user", "type": "address"},
        ],
        "name": "getUserReserveData",
        "outputs": [
            {"name": "currentATokenBalance", "type": "uint256"},
            {"name": "currentStableDebt", "type": "uint256"},
            {"name": "currentVariableDebt", "type": "uint256"},
            {"name": "principalStableDebt", "type": "uint256"},
            {"name": "scaledVariableDebt", "type": "uint256"},
            {"name": "stableBorrowRate", "type": "uint256"},
            {"name": "liquidityRate", "type": "uint256"},
            {"name": "stableRateLastUpdated", "type": "uint40"},
            {"name": "usageAsCollateralEnabled", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "asset", "type": "address"}],
        "name": "getReserveConfigurationData",
        "outputs": [
            {"name": "decimals", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "liquidationThreshold", "type": "uint256"},
            {"name": "liquidationBonus", "type": "uint256"},
            {"name": "reserveFactor", "type": "uint256"},
            {"name": "usageAsCollateralEnabled", "type": "bool"},
            {"name": "borrowingEnabled", "type": "bool"},
            {"name": "stableBorrowRateEnabled", "type": "bool"},
            {"name": "isActive", "type": "bool"},
            {"name": "isFrozen", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "asset", "type": "address"}],
        "name": "getReserveTokensAddresses",
        "outputs": [
            {"name": "aTokenAddress", "type": "address"},
            {"name": "stableDebtTokenAddress", "type": "address"},
            {"name": "variableDebtTokenAddress", "type": "address"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getAllReservesTokens",
        "outputs": [
            {
                "components": [
                    {"name": "symbol", "type": "string"},
                    {"name": "tokenAddress", "type": "address"},
                ],
                "name": "",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Oracle - price feeds
ORACLE_ABI = [
    {
        "inputs": [{"name": "asset", "type": "address"}],
        "name": "getAssetPrice",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "assets", "type": "address[]"}],
        "name": "getAssetsPrices",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "BASE_CURRENCY_UNIT",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ERC20 - basic token info
ERC20_ABI = [
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# =============================================================================
# Protocol Parameters
# =============================================================================

# Aave uses 8 decimals for USD values (BASE_CURRENCY_UNIT = 10^8)
AAVE_USD_DECIMALS = 8
AAVE_BASE_CURRENCY_UNIT = 10**8

# Health factor has 18 decimals
HEALTH_FACTOR_DECIMALS = 18

# Ray (27 decimals) used for rates
RAY = 10**27

# Percentage factor (10000 = 100%)
PERCENTAGE_FACTOR = 10000

# Maximum close factor for liquidation (50%)
MAX_LIQUIDATION_CLOSE_FACTOR = 5000  # 50% in basis points

# Health factor threshold for 100% liquidation
HF_THRESHOLD_FULL_LIQUIDATION = 0.95

# =============================================================================
# Common Arbitrum Tokens
# =============================================================================

ARBITRUM_TOKENS = {
    "WETH": {
        "address": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        "decimals": 18,
        "symbol": "WETH",
    },
    "USDC": {
        "address": "0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        "decimals": 6,
        "symbol": "USDC",
    },
    "USDC.e": {
        "address": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
        "decimals": 6,
        "symbol": "USDC.e",
    },
    "USDT": {
        "address": "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
        "decimals": 6,
        "symbol": "USDT",
    },
    "DAI": {
        "address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
        "decimals": 18,
        "symbol": "DAI",
    },
    "WBTC": {
        "address": "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f",
        "decimals": 8,
        "symbol": "WBTC",
    },
    "ARB": {
        "address": "0x912CE59144191C1204E64559FE8253a0e49E6548",
        "decimals": 18,
        "symbol": "ARB",
    },
    "LINK": {
        "address": "0xf97f4df75117a78c1A5a0DBb814Af92458539FB4",
        "decimals": 18,
        "symbol": "LINK",
    },
}
