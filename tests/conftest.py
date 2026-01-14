"""Pytest configuration and fixtures for MEV Liquidation Phase A tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from mev_analysis.core.safe_mode import SafeMode


@pytest.fixture(autouse=True)
def setup_safe_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SAFE_MODE is enabled for all tests."""
    monkeypatch.setenv("SAFE_MODE", "true")
    monkeypatch.setenv("MAX_EV_CAP_ETH", "1.0")


@pytest.fixture
def reset_safe_mode() -> Generator[None, None, None]:
    """Reset SafeMode singleton between tests."""
    SafeMode.reset()
    yield
    SafeMode.reset()


@pytest.fixture
def temp_log_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def safe_mode_instance(reset_safe_mode: None) -> SafeMode:
    """Provide an initialized SafeMode instance."""
    return SafeMode.initialize()
