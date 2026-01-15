"""Core modules for MEV analysis system."""

from mev_analysis.core.logging import ExperimentLogger, LogEntry, verify_log_integrity
from mev_analysis.core.safe_mode import SafeMode, SafeModeError
from mev_analysis.core.opportunity_detector import (
    DetectorConfig,
    OpportunityDetector,
    filter_actionable_opportunities,
)
from mev_analysis.core.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    WindowResult,
    create_synthetic_positions,
)

__all__ = [
    # Safe mode
    "SafeMode",
    "SafeModeError",
    # Logging
    "ExperimentLogger",
    "LogEntry",
    "verify_log_integrity",
    # Opportunity detection
    "DetectorConfig",
    "OpportunityDetector",
    "filter_actionable_opportunities",
    # Backtest
    "BacktestConfig",
    "BacktestResult",
    "BacktestRunner",
    "WindowResult",
    "create_synthetic_positions",
]
