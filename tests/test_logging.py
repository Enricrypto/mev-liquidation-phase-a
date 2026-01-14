"""Tests for hash-chained logging infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mev_analysis.core.logging import ExperimentLogger, LogEntry, verify_log_integrity


class TestLogEntry:
    """Tests for LogEntry model."""

    def test_log_entry_creation(self) -> None:
        """Should create a valid log entry."""
        entry = LogEntry(
            experiment_id="test_exp",
            run_number=1,
            level="INFO",
            message="Test message",
        )
        assert entry.experiment_id == "test_exp"
        assert entry.run_number == 1
        assert entry.level == "INFO"
        assert entry.message == "Test message"

    def test_log_entry_hash_computation(self) -> None:
        """Should compute consistent hash."""
        entry = LogEntry(
            experiment_id="test_exp",
            run_number=1,
            level="INFO",
            message="Test message",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        hash1 = entry.compute_hash()
        hash2 = entry.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_log_entry_finalize(self) -> None:
        """Should finalize with hash."""
        entry = LogEntry(
            experiment_id="test_exp",
            run_number=1,
            level="INFO",
            message="Test message",
        )
        entry.finalize()
        assert entry.entry_hash is not None
        assert entry.entry_hash == entry.compute_hash()

    def test_different_entries_have_different_hashes(self) -> None:
        """Different entries should have different hashes."""
        entry1 = LogEntry(
            experiment_id="test_exp",
            run_number=1,
            level="INFO",
            message="Message 1",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        entry2 = LogEntry(
            experiment_id="test_exp",
            run_number=1,
            level="INFO",
            message="Message 2",
            timestamp="2024-01-01T00:00:00+00:00",
        )
        assert entry1.compute_hash() != entry2.compute_hash()


class TestExperimentLogger:
    """Tests for ExperimentLogger."""

    def test_create_experiment(self, temp_log_dir: Path) -> None:
        """Should create experiment logger."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        assert exp_logger.experiment_id.startswith("test_run_")
        assert exp_logger.run_number == 1

    def test_log_info(self, temp_log_dir: Path) -> None:
        """Should log info messages."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Test info message", {"key": "value"})

        # Check JSON log file exists and has entry
        json_files = list(temp_log_dir.glob("*.jsonl"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            entry_json = f.readline()
            entry = json.loads(entry_json)
            assert entry["level"] == "INFO"
            assert entry["message"] == "Test info message"
            assert entry["data"]["key"] == "value"

    def test_log_metric(self, temp_log_dir: Path) -> None:
        """Should log metrics."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.log_metric("ev_estimate", 0.05, {"position_id": "0x123"})

        json_files = list(temp_log_dir.glob("*.jsonl"))
        with open(json_files[0]) as f:
            entry = json.loads(f.readline())
            assert entry["level"] == "METRIC"
            assert entry["data"]["metric_name"] == "ev_estimate"
            assert entry["data"]["value"] == 0.05
            assert entry["data"]["position_id"] == "0x123"

    def test_log_event(self, temp_log_dir: Path) -> None:
        """Should log events."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.log_event("liquidation_detected", {"position": "0xabc"})

        json_files = list(temp_log_dir.glob("*.jsonl"))
        with open(json_files[0]) as f:
            entry = json.loads(f.readline())
            assert entry["level"] == "EVENT"
            assert entry["data"]["event_type"] == "liquidation_detected"

    def test_hash_chaining(self, temp_log_dir: Path) -> None:
        """Should chain hashes correctly."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Message 1")
        exp_logger.info("Message 2")
        exp_logger.info("Message 3")

        json_files = list(temp_log_dir.glob("*.jsonl"))
        with open(json_files[0]) as f:
            lines = f.readlines()

        entries = [json.loads(line) for line in lines]

        # First entry has no previous hash
        assert entries[0]["previous_hash"] is None

        # Second entry's previous_hash is first entry's hash
        assert entries[1]["previous_hash"] == entries[0]["entry_hash"]

        # Third entry's previous_hash is second entry's hash
        assert entries[2]["previous_hash"] == entries[1]["entry_hash"]

    def test_get_log_summary(self, temp_log_dir: Path) -> None:
        """Should return log summary."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Message 1")
        exp_logger.info("Message 2")

        summary = exp_logger.get_log_summary()
        assert summary["entry_count"] == 2
        assert summary["last_hash"] is not None


class TestLogIntegrityVerification:
    """Tests for log integrity verification."""

    def test_verify_valid_log(self, temp_log_dir: Path) -> None:
        """Should verify valid log file."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Message 1")
        exp_logger.info("Message 2")
        exp_logger.info("Message 3")

        json_files = list(temp_log_dir.glob("*.jsonl"))
        is_valid, errors = verify_log_integrity(json_files[0])

        assert is_valid is True
        assert len(errors) == 0

    def test_detect_tampered_hash(self, temp_log_dir: Path) -> None:
        """Should detect tampered entry hash."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Message 1")
        exp_logger.info("Message 2")

        json_files = list(temp_log_dir.glob("*.jsonl"))

        # Tamper with the file
        with open(json_files[0]) as f:
            lines = f.readlines()

        entry = json.loads(lines[0])
        entry["entry_hash"] = "tampered_hash"

        with open(json_files[0], "w") as f:
            f.write(json.dumps(entry) + "\n")
            f.write(lines[1])

        is_valid, errors = verify_log_integrity(json_files[0])
        assert is_valid is False
        assert any("Entry hash mismatch" in e for e in errors)

    def test_detect_broken_chain(self, temp_log_dir: Path) -> None:
        """Should detect broken hash chain."""
        exp_logger = ExperimentLogger.create_experiment("test_run", temp_log_dir)
        exp_logger.info("Message 1")
        exp_logger.info("Message 2")

        json_files = list(temp_log_dir.glob("*.jsonl"))

        # Break the chain
        with open(json_files[0]) as f:
            lines = f.readlines()

        entry = json.loads(lines[1])
        entry["previous_hash"] = "wrong_previous_hash"
        # Recompute entry hash with wrong previous
        entry_obj = LogEntry.model_validate(entry)
        entry["entry_hash"] = entry_obj.compute_hash()

        with open(json_files[0], "w") as f:
            f.write(lines[0])
            f.write(json.dumps(entry) + "\n")

        is_valid, errors = verify_log_integrity(json_files[0])
        assert is_valid is False
        assert any("Hash chain broken" in e for e in errors)
