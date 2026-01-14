"""MEV Liquidation Phase A - Research and Validation System.

This package provides tools for empirically investigating liquidation MEV
opportunities on Aave v3 (Arbitrum) in a safe, reproducible manner.

Phase A is research and validation only - no real capital exposure.
"""

__version__ = "0.1.0"
__author__ = "Enrique Ibarra"

from mev_analysis.core.safe_mode import SafeMode, SafeModeError

__all__ = ["SafeMode", "SafeModeError", "__version__"]
