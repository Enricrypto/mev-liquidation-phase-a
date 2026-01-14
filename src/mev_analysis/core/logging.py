"""Hash-chained logging infrastructure for Phase A.

This module provides structured, immutable, hash-chained logging for:
- Full traceability of all experiments
- Independent verification of log integrity
- Reproducibility support via experiment IDs and random seeds

All logs are hash-chained: each entry includes the hash of the previous entry,
making tampering detectable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    """A single hash-chained log entry."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    experiment_id: str
    run_number: int
    level: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str | None = None
    entry_hash: str | None = None

    def compute_hash(self) -> str:
        """Compute the hash of this entry (excluding entry_hash field)."""
        data_for_hash = self.model_dump(exclude={"entry_hash"})
        json_str = json.dumps(data_for_hash, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def finalize(self) -> LogEntry:
        """Finalize the entry by computing its hash."""
        self.entry_hash = self.compute_hash()
        return self


class ExperimentLogger:
    """Hash-chained logger for experiment tracking.

    Creates structured, immutable logs with hash chaining for integrity verification.

    Usage:
        exp_logger = ExperimentLogger.create_experiment("backtest_run")
        exp_logger.info("Starting simulation", {"seed": 42})
        exp_logger.log_metric("ev_estimate", 0.05, {"position_id": "0x123"})
    """

    def __init__(
        self,
        experiment_id: str,
        log_dir: Path,
        run_number: int = 1,
    ) -> None:
        """Initialize the experiment logger.

        Args:
            experiment_id: Unique identifier for this experiment.
            log_dir: Directory to store log files.
            run_number: Run number within this experiment.
        """
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.run_number = run_number
        self._previous_hash: str | None = None
        self._entry_count = 0

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up log files
        self._json_log_path = self.log_dir / f"{experiment_id}_run{run_number}.jsonl"
        self._setup_loguru()

    def _setup_loguru(self) -> None:
        """Configure loguru for this experiment."""
        # Remove default handler
        logger.remove()

        # Add console handler with custom format
        logger.add(
            sys.stderr,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[experiment_id]}</cyan> | "
                "{message}"
            ),
            level="DEBUG",
            filter=lambda record: record["extra"].get("experiment_id") == self.experiment_id,
        )

        # Add file handler for human-readable logs
        text_log_path = self.log_dir / f"{self.experiment_id}_run{self.run_number}.log"
        logger.add(
            text_log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
            level="DEBUG",
            filter=lambda record: record["extra"].get("experiment_id") == self.experiment_id,
        )

        # Bind experiment context
        self._logger = logger.bind(
            experiment_id=self.experiment_id,
            run_number=self.run_number,
        )

    @classmethod
    def create_experiment(
        cls,
        name: str,
        log_dir: str | Path | None = None,
        run_number: int = 1,
    ) -> ExperimentLogger:
        """Create a new experiment logger.

        Args:
            name: Human-readable name for the experiment.
            log_dir: Directory for logs (default: from env or ./logs).
            run_number: Run number within this experiment.

        Returns:
            Configured ExperimentLogger instance.
        """
        # Generate unique experiment ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        experiment_id = f"{name}_{timestamp}_{short_uuid}"

        # Determine log directory
        if log_dir is None:
            log_dir = Path(os.getenv("LOG_DIR", "logs"))
        else:
            log_dir = Path(log_dir)

        return cls(experiment_id, log_dir, run_number)

    def _create_entry(
        self,
        level: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> LogEntry:
        """Create a hash-chained log entry."""
        entry = LogEntry(
            experiment_id=self.experiment_id,
            run_number=self.run_number,
            level=level,
            message=message,
            data=data or {},
            previous_hash=self._previous_hash,
        )
        entry.finalize()
        self._previous_hash = entry.entry_hash
        self._entry_count += 1
        return entry

    def _write_json_entry(self, entry: LogEntry) -> None:
        """Write entry to JSON log file."""
        with open(self._json_log_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def debug(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Log a debug message."""
        entry = self._create_entry("DEBUG", message, data)
        self._write_json_entry(entry)
        self._logger.debug(message)

    def info(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Log an info message."""
        entry = self._create_entry("INFO", message, data)
        self._write_json_entry(entry)
        self._logger.info(message)

    def warning(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Log a warning message."""
        entry = self._create_entry("WARNING", message, data)
        self._write_json_entry(entry)
        self._logger.warning(message)

    def error(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Log an error message."""
        entry = self._create_entry("ERROR", message, data)
        self._write_json_entry(entry)
        self._logger.error(message)

    def log_metric(
        self,
        metric_name: str,
        value: float | int,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log a metric value with context.

        Args:
            metric_name: Name of the metric (e.g., 'ev_estimate', 'gas_used').
            value: The metric value.
            context: Additional context data.
        """
        data = {
            "metric_name": metric_name,
            "value": value,
            **(context or {}),
        }
        entry = self._create_entry("METRIC", f"{metric_name}={value}", data)
        self._write_json_entry(entry)
        self._logger.info(f"METRIC: {metric_name}={value}")

    def log_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
    ) -> None:
        """Log a structured event.

        Args:
            event_type: Type of event (e.g., 'liquidation_detected', 'bot_action').
            event_data: Event-specific data.
        """
        data = {
            "event_type": event_type,
            **event_data,
        }
        entry = self._create_entry("EVENT", f"Event: {event_type}", data)
        self._write_json_entry(entry)
        self._logger.info(f"EVENT: {event_type}")

    def get_log_summary(self) -> dict[str, Any]:
        """Get a summary of the current log state."""
        return {
            "experiment_id": self.experiment_id,
            "run_number": self.run_number,
            "entry_count": self._entry_count,
            "last_hash": self._previous_hash,
            "json_log_path": str(self._json_log_path),
        }


def verify_log_integrity(log_path: Path) -> tuple[bool, list[str]]:
    """Verify the integrity of a hash-chained log file.

    Args:
        log_path: Path to the JSONL log file.

    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors: list[str] = []
    previous_hash: str | None = None

    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = LogEntry.model_validate_json(line)

                # Verify previous hash chain
                if entry.previous_hash != previous_hash:
                    errors.append(
                        f"Line {line_num}: Hash chain broken. "
                        f"Expected previous_hash={previous_hash}, "
                        f"got {entry.previous_hash}"
                    )

                # Verify entry hash
                computed_hash = entry.compute_hash()
                if entry.entry_hash != computed_hash:
                    errors.append(
                        f"Line {line_num}: Entry hash mismatch. "
                        f"Expected {computed_hash}, got {entry.entry_hash}"
                    )

                previous_hash = entry.entry_hash

            except Exception as e:
                errors.append(f"Line {line_num}: Parse error - {e}")

    return len(errors) == 0, errors
