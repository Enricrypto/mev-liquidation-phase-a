"""Tests for position loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mev_analysis.data.models import KnownPosition
from mev_analysis.data.position_loader import (
    save_positions_to_csv,
    save_positions_to_json,
)


class TestPositionLoaderIO:
    """Tests for position loader file I/O (no RPC needed)."""

    @pytest.fixture
    def sample_positions(self) -> list[KnownPosition]:
        """Sample known positions."""
        return [
            KnownPosition(
                user_address="0x1234567890abcdef1234567890abcdef12345678",
                block_number=12345678,
                label="whale_1",
                source="historical",
            ),
            KnownPosition(
                user_address="0xabcdef1234567890abcdef1234567890abcdef12",
                block_number=None,
                label="active_user",
                source="subgraph",
            ),
            KnownPosition(
                user_address="0x9876543210fedcba9876543210fedcba98765432",
                block_number=12345000,
                label="",
                source="manual",
            ),
        ]

    def test_save_and_load_csv(
        self, tmp_path: Path, sample_positions: list[KnownPosition]
    ) -> None:
        """Should save and load positions from CSV."""
        csv_path = tmp_path / "positions.csv"

        # Save
        save_positions_to_csv(sample_positions, csv_path)
        assert csv_path.exists()

        # Verify content
        content = csv_path.read_text()
        assert "user_address,block_number,label,source" in content
        assert "0x1234567890abcdef1234567890abcdef12345678" in content
        assert "whale_1" in content
        assert "12345678" in content

    def test_save_and_load_json(
        self, tmp_path: Path, sample_positions: list[KnownPosition]
    ) -> None:
        """Should save and load positions from JSON."""
        json_path = tmp_path / "positions.json"

        # Save
        save_positions_to_json(sample_positions, json_path)
        assert json_path.exists()

        # Verify content
        with open(json_path) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["user_address"] == "0x1234567890abcdef1234567890abcdef12345678"
        assert data[0]["block_number"] == 12345678
        assert data[0]["label"] == "whale_1"

    def test_csv_roundtrip(
        self, tmp_path: Path, sample_positions: list[KnownPosition]
    ) -> None:
        """Should maintain data integrity through CSV save/load cycle."""
        csv_path = tmp_path / "roundtrip.csv"
        save_positions_to_csv(sample_positions, csv_path)

        # Manual load to verify (PositionLoader.load_from_csv needs AaveV3Client)
        import csv as csv_module

        loaded = []
        with open(csv_path, newline="") as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                block_num = row.get("block_number", "").strip()
                loaded.append(
                    KnownPosition(
                        user_address=row["user_address"],
                        block_number=int(block_num) if block_num else None,
                        label=row.get("label", ""),
                        source=row.get("source", "csv"),
                    )
                )

        assert len(loaded) == len(sample_positions)
        assert loaded[0].user_address == sample_positions[0].user_address
        assert loaded[0].block_number == sample_positions[0].block_number

    def test_json_roundtrip(
        self, tmp_path: Path, sample_positions: list[KnownPosition]
    ) -> None:
        """Should maintain data integrity through JSON save/load cycle."""
        json_path = tmp_path / "roundtrip.json"
        save_positions_to_json(sample_positions, json_path)

        # Load and validate
        with open(json_path) as f:
            data = json.load(f)

        loaded = [KnownPosition.model_validate(item) for item in data]

        assert len(loaded) == len(sample_positions)
        for original, loaded_pos in zip(sample_positions, loaded):
            assert loaded_pos.user_address == original.user_address
            assert loaded_pos.block_number == original.block_number
            assert loaded_pos.label == original.label
            assert loaded_pos.source == original.source

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """Should create parent directories if needed."""
        nested_path = tmp_path / "a" / "b" / "c" / "positions.json"
        positions = [
            KnownPosition(
                user_address="0x1234567890abcdef1234567890abcdef12345678"
            )
        ]

        save_positions_to_json(positions, nested_path)
        assert nested_path.exists()

    def test_handles_empty_fields(self, tmp_path: Path) -> None:
        """Should handle positions with empty optional fields."""
        positions = [
            KnownPosition(
                user_address="0x1234567890abcdef1234567890abcdef12345678",
                # block_number is None
                # label defaults to ""
                # source defaults to "manual"
            )
        ]

        csv_path = tmp_path / "empty_fields.csv"
        save_positions_to_csv(positions, csv_path)

        content = csv_path.read_text()
        # Should have empty block_number field
        lines = content.strip().split("\n")
        assert len(lines) == 2  # Header + 1 row
        # The row should contain the address and empty block_number
        assert "0x1234567890abcdef1234567890abcdef12345678" in lines[1]
