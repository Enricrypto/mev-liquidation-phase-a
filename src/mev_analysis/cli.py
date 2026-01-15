"""Command-line interface for MEV liquidation analysis.

Provides CLI entry points for:
- Running backtests on position data
- Running simulations on opportunities
- Generating sample data for testing

All operations run in SAFE_MODE (research only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from mev_analysis.core.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
    create_synthetic_positions,
)
from mev_analysis.core.logging import ExperimentLogger
from mev_analysis.core.opportunity_detector import DetectorConfig, OpportunityDetector
from mev_analysis.core.safe_mode import SafeMode
from mev_analysis.data.models import MarketConditions, UserPosition
from mev_analysis.data.position_loader import PositionLoader
from mev_analysis.simulation.engine import SimulationConfig, SimulationEngine


def _setup_logger(output_dir: Path, verbose: bool) -> ExperimentLogger:
    """Set up experiment logger."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentLogger(
        experiment_id=f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_dir=log_dir,
    )


def _format_decimal(d: Decimal) -> str:
    """Format decimal for display."""
    return f"{float(d):.6f}"


def _result_to_dict(result: BacktestResult) -> dict[str, Any]:
    """Convert BacktestResult to JSON-serializable dict."""
    return {
        "backtest_id": result.backtest_id,
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "total_positions_scanned": result.total_positions_scanned,
        "total_opportunities_detected": result.total_opportunities_detected,
        "total_opportunities_simulated": result.total_opportunities_simulated,
        "overall_mean_ev_eth": _format_decimal(result.overall_mean_ev_eth),
        "overall_ev_ci_lower_95": _format_decimal(result.overall_ev_ci_lower_95),
        "overall_ev_ci_upper_95": _format_decimal(result.overall_ev_ci_upper_95),
        "overall_mean_capture_prob": _format_decimal(result.overall_mean_capture_prob),
        "overall_capture_ci_lower_95": _format_decimal(result.overall_capture_ci_lower_95),
        "overall_capture_ci_upper_95": _format_decimal(result.overall_capture_ci_upper_95),
        "hypothesis_results": result.hypothesis_results,
        "meets_sample_size": result.meets_sample_size,
        "meets_capture_threshold": result.meets_capture_threshold,
        "meets_ev_threshold": result.meets_ev_threshold,
        "num_windows": len(result.windows),
        "windows": [
            {
                "window_id": w.window_id,
                "start_block": w.start_block,
                "end_block": w.end_block,
                "num_positions_scanned": w.num_positions_scanned,
                "num_opportunities_detected": w.num_opportunities_detected,
                "mean_ev_eth": _format_decimal(w.mean_ev_eth),
                "mean_capture_probability": _format_decimal(w.mean_capture_probability),
                "duration_ms": w.duration_ms,
            }
            for w in result.windows
        ],
    }


def run_backtest() -> None:
    """CLI entry point for running backtests."""
    parser = argparse.ArgumentParser(
        prog="mev-backtest",
        description="Run backtest simulation on position data (research only)",
    )

    # Input options
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--positions",
        type=Path,
        help="Path to positions file (CSV or JSON)",
    )
    input_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic positions for testing",
    )
    input_group.add_argument(
        "--num-positions",
        type=int,
        default=100,
        help="Number of synthetic positions (default: 100)",
    )
    input_group.add_argument(
        "--num-liquidatable",
        type=int,
        default=20,
        help="Number of liquidatable positions (default: 20)",
    )

    # Backtest configuration
    config_group = parser.add_argument_group("configuration")
    config_group.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size in blocks (default: 1000)",
    )
    config_group.add_argument(
        "--window-stride",
        type=int,
        default=500,
        help="Window stride in blocks (default: 500)",
    )
    config_group.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of simulation iterations (default: 1000)",
    )
    config_group.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds for reproducibility (default: 10)",
    )
    config_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    config_group.add_argument(
        "--min-profit-eth",
        type=float,
        default=0.01,
        help="Minimum profit threshold in ETH (default: 0.01)",
    )

    # Market conditions
    market_group = parser.add_argument_group("market")
    market_group.add_argument(
        "--gas-price",
        type=float,
        default=0.1,
        help="Gas price in gwei (default: 0.1)",
    )
    market_group.add_argument(
        "--eth-price",
        type=float,
        default=2000.0,
        help="ETH price in USD (default: 2000)",
    )

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    output_group.add_argument(
        "--output-json",
        type=Path,
        help="Path to save results as JSON",
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Verify safe mode - will raise SafeModeError if SAFE_MODE env not set
    try:
        safe_mode = SafeMode()
    except Exception as e:
        print(f"ERROR: SAFE_MODE check failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("MEV Liquidation Backtest (RESEARCH MODE)")
    print("=" * 60)
    print(f"Safe mode: enabled (max EV cap: {safe_mode.config.max_ev_cap_eth} ETH)")
    print()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(args.output_dir, args.verbose)

    # Load or generate positions
    if args.synthetic:
        print(f"Generating {args.num_positions} synthetic positions "
              f"({args.num_liquidatable} liquidatable)...")
        positions = create_synthetic_positions(
            num_positions=args.num_positions,
            num_liquidatable=args.num_liquidatable,
            seed=args.seed,
        )
    elif args.positions:
        print(f"Loading positions from {args.positions}...")
        loader = PositionLoader()
        positions = loader.load(args.positions)
    else:
        print("ERROR: Must specify --positions or --synthetic", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(positions)} positions")

    # Create market conditions
    market_conditions = MarketConditions(
        block_number=positions[0].block_number if positions else 1000000,
        timestamp=datetime.now(timezone.utc),
        gas_price_gwei=Decimal(str(args.gas_price)),
        eth_price_usd=Decimal(str(args.eth_price)),
    )

    # Configure backtest
    config = BacktestConfig(
        window_size_blocks=args.window_size,
        window_stride_blocks=args.window_stride,
        num_simulation_iterations=args.num_iterations,
        num_seeds=args.num_seeds,
        base_seed=args.seed,
        min_profit_eth=Decimal(str(args.min_profit_eth)),
    )

    print()
    print("Configuration:")
    print(f"  Window size: {config.window_size_blocks} blocks")
    print(f"  Window stride: {config.window_stride_blocks} blocks")
    print(f"  Iterations per opportunity: {config.num_simulation_iterations}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Min profit threshold: {config.min_profit_eth} ETH")
    print()

    # Run backtest
    print("Running backtest...")
    runner = BacktestRunner(config=config, logger=logger)
    result = runner.run(positions, market_conditions)

    # Display results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Backtest ID: {result.backtest_id}")
    print(f"Duration: {result.completed_at - result.started_at}")
    print()
    print("Summary:")
    print(f"  Positions scanned: {result.total_positions_scanned}")
    print(f"  Opportunities detected: {result.total_opportunities_detected}")
    print(f"  Simulations run: {result.total_opportunities_simulated}")
    print()
    print("Expected Value (ETH):")
    print(f"  Mean: {_format_decimal(result.overall_mean_ev_eth)}")
    print(f"  95% CI: [{_format_decimal(result.overall_ev_ci_lower_95)}, "
          f"{_format_decimal(result.overall_ev_ci_upper_95)}]")
    print()
    print("Capture Probability:")
    print(f"  Mean: {float(result.overall_mean_capture_prob) * 100:.2f}%")
    print(f"  95% CI: [{float(result.overall_capture_ci_lower_95) * 100:.2f}%, "
          f"{float(result.overall_capture_ci_upper_95) * 100:.2f}%]")
    print()
    print("Validation:")
    print(f"  Meets sample size (≥30): {'✓' if result.meets_sample_size else '✗'}")
    print(f"  Meets capture threshold (≥3%): {'✓' if result.meets_capture_threshold else '✗'}")
    print(f"  Meets EV threshold (≥0.01 ETH): {'✓' if result.meets_ev_threshold else '✗'}")
    print()
    print("Hypothesis Testing:")
    for key, value in result.hypothesis_results.items():
        if isinstance(value, dict):
            print(f"  {key}: {value.get('result', 'N/A')}")
        else:
            print(f"  {key}: {value}")
    print()

    # Save results
    if args.output_json:
        result_dict = _result_to_dict(result)
        with open(args.output_json, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to {args.output_json}")

    # Auto-save to output directory
    auto_save_path = args.output_dir / f"{result.backtest_id}.json"
    result_dict = _result_to_dict(result)
    with open(auto_save_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"Results auto-saved to {auto_save_path}")

    print()
    print("Backtest complete.")


def run_simulation() -> None:
    """CLI entry point for running simulations on detected opportunities."""
    parser = argparse.ArgumentParser(
        prog="mev-simulate",
        description="Run Monte Carlo simulation on liquidation opportunities",
    )

    # Input options
    input_group = parser.add_argument_group("input")
    input_group.add_argument(
        "--positions",
        type=Path,
        help="Path to positions file (CSV or JSON)",
    )
    input_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic positions for testing",
    )
    input_group.add_argument(
        "--num-positions",
        type=int,
        default=50,
        help="Number of synthetic positions (default: 50)",
    )
    input_group.add_argument(
        "--num-liquidatable",
        type=int,
        default=10,
        help="Number of liquidatable positions (default: 10)",
    )

    # Simulation configuration
    config_group = parser.add_argument_group("configuration")
    config_group.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of simulation iterations (default: 1000)",
    )
    config_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    config_group.add_argument(
        "--min-profit-eth",
        type=float,
        default=0.01,
        help="Minimum profit threshold in ETH (default: 0.01)",
    )

    # Market conditions
    market_group = parser.add_argument_group("market")
    market_group.add_argument(
        "--gas-price",
        type=float,
        default=0.1,
        help="Gas price in gwei (default: 0.1)",
    )
    market_group.add_argument(
        "--eth-price",
        type=float,
        default=2000.0,
        help="ETH price in USD (default: 2000)",
    )

    # Output options
    output_group = parser.add_argument_group("output")
    output_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    output_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Verify safe mode - will raise SafeModeError if SAFE_MODE env not set
    try:
        safe_mode = SafeMode()
    except Exception as e:
        print(f"ERROR: SAFE_MODE check failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("MEV Liquidation Simulation (RESEARCH MODE)")
    print("=" * 60)
    print(f"Safe mode: enabled (max EV cap: {safe_mode.config.max_ev_cap_eth} ETH)")
    print()

    # Setup
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(args.output_dir, args.verbose)

    # Load or generate positions
    if args.synthetic:
        print(f"Generating {args.num_positions} synthetic positions "
              f"({args.num_liquidatable} liquidatable)...")
        positions = create_synthetic_positions(
            num_positions=args.num_positions,
            num_liquidatable=args.num_liquidatable,
            seed=args.seed,
        )
    elif args.positions:
        print(f"Loading positions from {args.positions}...")
        loader = PositionLoader()
        positions = loader.load(args.positions)
    else:
        print("ERROR: Must specify --positions or --synthetic", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(positions)} positions")

    # Create market conditions
    market_conditions = MarketConditions(
        block_number=positions[0].block_number if positions else 1000000,
        timestamp=datetime.now(timezone.utc),
        gas_price_gwei=Decimal(str(args.gas_price)),
        eth_price_usd=Decimal(str(args.eth_price)),
    )

    # Detect opportunities
    print()
    print("Detecting opportunities...")
    detector_config = DetectorConfig(
        min_profit_eth=Decimal(str(args.min_profit_eth)),
    )
    detector = OpportunityDetector(detector_config, logger)
    opportunities = list(detector.detect(positions, market_conditions))
    print(f"Detected {len(opportunities)} opportunities")

    if not opportunities:
        print("No opportunities to simulate.")
        return

    # Simulate opportunities
    print()
    print(f"Running simulation ({args.num_iterations} iterations per opportunity)...")

    sim_config = SimulationConfig(
        num_iterations=args.num_iterations,
        base_seed=args.seed,
    )
    engine = SimulationEngine(config=sim_config, logger=logger)

    results = []
    for i, opp in enumerate(opportunities, 1):
        print(f"  Simulating opportunity {i}/{len(opportunities)}: {opp.opportunity_id}")
        result = engine.simulate(opp, market_conditions)
        results.append(result)

        if args.verbose:
            print(f"    Capture probability: {float(result.capture_probability) * 100:.2f}%")
            print(f"    Mean EV: {_format_decimal(result.mean_ev_eth)} ETH")

    # Display summary
    print()
    print("=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    total_capture = sum(r.capture_probability for r in results) / len(results)
    total_ev = sum(r.mean_ev_eth for r in results)
    mean_ev = total_ev / len(results)

    print(f"Opportunities simulated: {len(results)}")
    print(f"Average capture probability: {float(total_capture) * 100:.2f}%")
    print(f"Total expected EV: {_format_decimal(total_ev)} ETH")
    print(f"Mean EV per opportunity: {_format_decimal(mean_ev)} ETH")
    print()

    # Failure analysis
    all_failures: dict[str, int] = {}
    for r in results:
        for reason, count in r.failure_reasons.items():
            all_failures[reason] = all_failures.get(reason, 0) + count

    if all_failures:
        print("Failure breakdown:")
        for reason, count in sorted(all_failures.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print()
    print("Simulation complete.")


def main() -> None:
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mev-analysis",
        description="MEV liquidation analysis toolkit (research only)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run backtest simulation",
    )
    backtest_parser.set_defaults(func=run_backtest)

    # Simulate command
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run Monte Carlo simulation",
    )
    simulate_parser.set_defaults(func=run_simulation)

    args = parser.parse_args()

    if args.command:
        args.func()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
