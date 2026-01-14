"""Position loader for known addresses.

Supports loading positions from:
- CSV files
- JSON files
- Python lists

Designed for Phase A hybrid approach: start with curated known positions,
then expand to full discovery.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterator

from mev_analysis.data.models import KnownPosition, UserPosition
from mev_analysis.data.aave_v3 import AaveV3Client
from mev_analysis.core.logging import ExperimentLogger


class PositionLoader:
    """Load and fetch positions from known addresses.

    Usage:
        loader = PositionLoader(aave_client, logger)
        positions = loader.load_from_csv("positions.csv")
        for position in loader.fetch_positions(positions, block_number=12345):
            print(position.health_factor)
    """

    def __init__(
        self,
        aave_client: AaveV3Client,
        logger: ExperimentLogger | None = None,
    ) -> None:
        """Initialize position loader.

        Args:
            aave_client: Configured AaveV3Client instance.
            logger: Optional experiment logger for tracking.
        """
        self.aave_client = aave_client
        self.logger = logger

    def load_from_csv(self, file_path: str | Path) -> list[KnownPosition]:
        """Load known positions from CSV file.

        Expected CSV format:
            user_address,block_number,label,source
            0x123...,12345,whale_1,historical
            0x456...,,active_user,subgraph

        Args:
            file_path: Path to CSV file.

        Returns:
            List of KnownPosition objects.
        """
        positions: list[KnownPosition] = []
        path = Path(file_path)

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle empty block_number
                block_num = row.get("block_number", "").strip()
                block_number = int(block_num) if block_num else None

                position = KnownPosition(
                    user_address=row["user_address"].strip(),
                    block_number=block_number,
                    label=row.get("label", "").strip(),
                    source=row.get("source", "csv").strip(),
                )
                positions.append(position)

        if self.logger:
            self.logger.info(
                f"Loaded {len(positions)} positions from CSV",
                {"file": str(path), "count": len(positions)},
            )

        return positions

    def load_from_json(self, file_path: str | Path) -> list[KnownPosition]:
        """Load known positions from JSON file.

        Expected JSON format:
            [
                {"user_address": "0x123...", "block_number": 12345, "label": "whale"},
                {"user_address": "0x456...", "label": "active_user"}
            ]

        Args:
            file_path: Path to JSON file.

        Returns:
            List of KnownPosition objects.
        """
        path = Path(file_path)

        with open(path) as f:
            data = json.load(f)

        positions = [KnownPosition.model_validate(item) for item in data]

        if self.logger:
            self.logger.info(
                f"Loaded {len(positions)} positions from JSON",
                {"file": str(path), "count": len(positions)},
            )

        return positions

    def load_from_list(
        self,
        addresses: list[str],
        block_number: int | None = None,
        source: str = "manual",
    ) -> list[KnownPosition]:
        """Create known positions from a list of addresses.

        Args:
            addresses: List of user wallet addresses.
            block_number: Optional block number (applied to all).
            source: Source label for tracking.

        Returns:
            List of KnownPosition objects.
        """
        positions = [
            KnownPosition(
                user_address=addr,
                block_number=block_number,
                source=source,
            )
            for addr in addresses
        ]

        if self.logger:
            self.logger.info(
                f"Created {len(positions)} positions from list",
                {"count": len(positions), "source": source},
            )

        return positions

    def fetch_positions(
        self,
        known_positions: list[KnownPosition],
        block_number: int | None = None,
        skip_empty: bool = True,
    ) -> Iterator[UserPosition]:
        """Fetch full position data for known addresses.

        Args:
            known_positions: List of known positions to fetch.
            block_number: Override block number (uses position's block if None).
            skip_empty: Skip positions with no collateral or debt.

        Yields:
            UserPosition objects with full data.
        """
        for known in known_positions:
            # Determine block to query
            query_block = block_number or known.block_number or "latest"

            try:
                position = self.aave_client.get_user_position(
                    known.user_address,
                    block_identifier=query_block,
                )

                # Skip empty positions if requested
                if skip_empty and not position.collaterals and not position.debts:
                    if self.logger:
                        self.logger.debug(
                            f"Skipping empty position: {known.user_address}",
                            {"address": known.user_address, "block": query_block},
                        )
                    continue

                if self.logger:
                    self.logger.log_event(
                        "position_fetched",
                        {
                            "user_address": position.user_address,
                            "block_number": position.block_number,
                            "health_factor": str(position.health_factor),
                            "collateral_count": len(position.collaterals),
                            "debt_count": len(position.debts),
                            "is_liquidatable": position.is_liquidatable,
                            "source": known.source,
                            "label": known.label,
                        },
                    )

                yield position

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Failed to fetch position: {known.user_address}",
                        {
                            "address": known.user_address,
                            "block": str(query_block),
                            "error": str(e),
                        },
                    )
                # Continue to next position
                continue

    def fetch_liquidatable_positions(
        self,
        known_positions: list[KnownPosition],
        block_number: int | None = None,
    ) -> Iterator[UserPosition]:
        """Fetch only liquidatable positions.

        Args:
            known_positions: List of known positions to check.
            block_number: Override block number.

        Yields:
            UserPosition objects that are liquidatable.
        """
        for position in self.fetch_positions(known_positions, block_number):
            if position.is_liquidatable:
                if self.logger:
                    self.logger.log_event(
                        "liquidatable_position_found",
                        {
                            "user_address": position.user_address,
                            "block_number": position.block_number,
                            "health_factor": str(position.health_factor),
                            "total_collateral_usd": str(position.total_collateral_usd),
                            "total_debt_usd": str(position.total_debt_usd),
                        },
                    )
                yield position


def save_positions_to_csv(
    positions: list[KnownPosition],
    file_path: str | Path,
) -> None:
    """Save known positions to CSV file.

    Args:
        positions: List of positions to save.
        file_path: Output file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["user_address", "block_number", "label", "source"]
        )
        writer.writeheader()
        for pos in positions:
            writer.writerow(
                {
                    "user_address": pos.user_address,
                    "block_number": pos.block_number or "",
                    "label": pos.label,
                    "source": pos.source,
                }
            )


def save_positions_to_json(
    positions: list[KnownPosition],
    file_path: str | Path,
) -> None:
    """Save known positions to JSON file.

    Args:
        positions: List of positions to save.
        file_path: Output file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [pos.model_dump() for pos in positions]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
