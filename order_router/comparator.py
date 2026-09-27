"""
comparator.py — Strategy comparison harness driven by real AAPL bar data.

Replays N windows of price bars from AAPL_1min_2024-2026.csv.  Each window:
  1. Creates three fresh Exchange instances (ALPHA, BETA, GAMMA).
  2. Runs all four routing strategies against the same market conditions.
  3. Collects RouterResult for each strategy.

After all windows, aggregates results into a ComparisonReport and prints a
formatted summary table.

Run as a module:
  uv run python -m order_router.comparator
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from .exchange import Exchange, VENUE_CONFIGS
from .models import Order, OrderSide, OrderType
from .price_feed import PriceFeed, Bar
from .router import OrderRouter, RouterResult
from .routing import (
    BestPriceStrategy,
    TWAPStrategy,
    VWAPStrategy,
    SmartStrategy,
    RoutingStrategy,
)


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------

@dataclass
class ComparisonReport:
    """
    Aggregated comparison of all strategies over the replay window.

    Attributes
    ----------
    bars_replayed : Total number of bars consumed across all windows.
    num_windows   : Number of distinct windows replayed.
    order_qty     : Parent order quantity used in each window.
    results       : strategy_name → list of RouterResult (one per window).
    summary       : strategy_name → dict of aggregated metric floats.
    """

    bars_replayed: int
    num_windows: int
    order_qty: float
    results: Dict[str, List[RouterResult]] = field(default_factory=dict)
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StrategyComparator
# ---------------------------------------------------------------------------

class StrategyComparator:
    """
    Replay-based strategy comparison harness.

    Parameters
    ----------
    csv_path    : Path to AAPL_1min_2024-2026.csv.
    num_windows : Number of bar-windows to replay.  Default 20.
    window_size : Bars per window (= number of TWAP/VWAP slices).  Default 5.
    order_qty   : Shares per synthetic parent order.  Default 5000.
                  Using a large quantity stresses venue depth and makes
                  fill-rate differences between strategies visible.
    side        : BUY or SELL.  Default BUY.
    """

    def __init__(
        self,
        csv_path: str,
        num_windows: int = 20,
        window_size: int = 5,
        order_qty: float = 5000.0,
        side: OrderSide = OrderSide.BUY,
    ) -> None:
        self.csv_path = csv_path
        self.num_windows = num_windows
        self.window_size = window_size
        self.order_qty = order_qty
        self.side = side

        self._strategies: List[RoutingStrategy] = [
            BestPriceStrategy(),
            TWAPStrategy(num_slices=window_size),
            VWAPStrategy(num_slices=window_size),
            SmartStrategy(),
        ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> ComparisonReport:
        """
        Execute the full comparison and return a ComparisonReport.
        """
        feed = PriceFeed(self.csv_path)
        bars = list(feed)

        all_results: Dict[str, List[RouterResult]] = {
            s.name: [] for s in self._strategies
        }
        windows_completed = 0

        for w in range(self.num_windows):
            start = w * self.window_size
            window: List[Bar] = bars[start: start + self.window_size]
            if len(window) < self.window_size:
                break  # not enough bars left

            for strategy in self._strategies:
                # Give each strategy its own fresh, identically-seeded venues
                venues = {
                    name: Exchange(name, cfg)
                    for name, cfg in VENUE_CONFIGS.items()
                }

                order = Order(
                    symbol="AAPL",
                    side=self.side,
                    order_type=OrderType.MARKET,
                    quantity=self.order_qty,
                )

                router = OrderRouter(venues=venues, strategy=strategy)
                result = router.submit(order, bars=window)
                all_results[strategy.name].append(result)

            windows_completed += 1

        summary = self._build_summary(all_results, windows_completed)

        return ComparisonReport(
            bars_replayed=windows_completed * self.window_size,
            num_windows=windows_completed,
            order_qty=self.order_qty,
            results=all_results,
            summary=summary,
        )

    def print_report(self, report: ComparisonReport) -> None:
        """Print a formatted comparison table to stdout."""
        self._print_header(report)
        self._print_table(report.summary)
        self._print_footer(report)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        all_results: Dict[str, List[RouterResult]],
        num_windows: int,
    ) -> Dict[str, Dict[str, float]]:
        summary = {}
        for name, results in all_results.items():
            filled = [r for r in results if r.total_filled > 0]
            if not filled:
                summary[name] = {}
                continue

            total_requested = self.order_qty * len(results)
            total_filled = sum(r.total_filled for r in results)

            summary[name] = {
                "avg_fill_price":    round(
                    sum(r.avg_fill_price for r in filled) / len(filled), 4
                ),
                "total_fees":        round(sum(r.total_fees for r in filled), 2),
                "avg_slippage_bps":  round(
                    sum(r.total_slippage_bps for r in filled) / len(filled), 4
                ),
                "fill_rate_pct":     round(
                    100.0 * total_filled / total_requested, 2
                ),
                "avg_latency_ms":    round(
                    sum(r.execution_time_ms for r in results) / len(results), 1
                ),
                "windows":           num_windows,
            }
        return summary

    @staticmethod
    def _print_header(report: ComparisonReport) -> None:
        print()
        print("=" * 76)
        print("  ORDER ROUTER — STRATEGY COMPARISON REPORT")
        print("=" * 76)
        print(f"  Symbol       : AAPL")
        print(f"  Order size   : {report.order_qty:,.0f} shares (market order, BUY)")
        print(f"  Windows      : {report.num_windows}  ×  {report.bars_replayed // max(report.num_windows, 1)} bars")
        print(f"  Total bars   : {report.bars_replayed:,}")
        print(f"  Exchanges    : ALPHA (2 bps, 1 ms) | BETA (1 bps, 5 ms) | GAMMA (0.5 bps, 15 ms)")
        print("=" * 76)

    @staticmethod
    def _print_table(summary: Dict[str, Dict[str, float]]) -> None:
        col_w = 13
        headers = [
            "Strategy", "Fill Rate%", "Avg Price", "Fees $",
            "Slip (bps)", "Latency ms",
        ]
        header_row = (
            f"{'Strategy':<14}"
            f"{'Fill Rate%':>{col_w}}"
            f"{'Avg Price':>{col_w}}"
            f"{'Fees $':>{col_w}}"
            f"{'Slip (bps)':>{col_w}}"
            f"{'Latency ms':>{col_w}}"
        )
        separator = "-" * len(header_row)

        print()
        print(header_row)
        print(separator)

        for name, m in summary.items():
            if not m:
                print(f"{name:<14}{'N/A':>{col_w}}")
                continue
            print(
                f"{name:<14}"
                f"{m['fill_rate_pct']:>{col_w}.2f}"
                f"{m['avg_fill_price']:>{col_w}.4f}"
                f"{m['total_fees']:>{col_w}.2f}"
                f"{m['avg_slippage_bps']:>{col_w}.4f}"
                f"{m['avg_latency_ms']:>{col_w}.1f}"
            )

        print(separator)
        print()
        print("  Notes:")
        print("  • Fill Rate%   : shares filled / shares requested (higher = better)")
        print("  • Avg Price    : volume-weighted average fill price (lower = better for BUY)")
        print("  • Fees $       : total exchange fees paid across all windows")
        print("  • Slip (bps)   : avg fill slippage vs VWAP reference (lower = better)")
        print("  • Latency ms   : cumulative simulated round-trip latency")
        print()

    @staticmethod
    def _print_footer(report: ComparisonReport) -> None:
        print("=" * 76)
        print("  Strategy summary:")
        print("  BestPrice — greedy single-venue; lowest latency, lowest fill rate")
        print("  TWAP      — equal time slices; spreads over N bars for full fills")
        print("  VWAP      — volume-weighted slices; tracks market rhythm")
        print("  Smart     — cross-venue sweep; best fill rate at single time step")
        print("=" * 76)
        print()


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(here, "AAPL_1min_2024-2026.csv")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    comparator = StrategyComparator(
        csv_path=csv_path,
        num_windows=20,
        window_size=5,
        order_qty=5000.0,
    )
    report = comparator.run()
    comparator.print_report(report)


if __name__ == "__main__":
    main()
