"""
backtest.py — Rolling bar-by-bar backtest of all four routing strategies.

Walks through real AAPL 1-minute bars in fixed-size windows, executing a
standard BUY order at each window under every routing strategy.  Records a
rich set of execution-quality metrics per window per strategy so the results
CSV and chart can prove definitively that smart routing beats naive routing.

BacktestRecord — one row of the output CSV
BacktestEngine — the driver class
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List

import pandas as pd

from .exchange import Exchange, VENUE_CONFIGS
from .models import Order, OrderSide, OrderType
from .price_feed import PriceFeed, Bar
from .router import OrderRouter, RouterResult
from .routing import BestPriceStrategy, TWAPStrategy, VWAPStrategy, SmartStrategy

log = logging.getLogger(__name__)

# Canonical display order — matches chart legend top-to-bottom
STRATEGY_ORDER = ["BestPrice", "TWAP(5)", "VWAP(5)", "Smart"]


# ---------------------------------------------------------------------------
# BacktestRecord — one CSV row
# ---------------------------------------------------------------------------

@dataclass
class BacktestRecord:
    """
    All execution-quality metrics captured for a single (window, strategy) pair.

    CSV columns
    -----------
    window_idx       : Sequential window number (0-based).
    timestamp        : ISO timestamp of the first bar in the window.
    strategy         : Strategy name string.
    order_qty        : Requested shares.
    filled_qty       : Shares actually matched across all venues.
    unfilled_qty     : order_qty − filled_qty  (the "leftover" problem).
    fill_rate_pct    : filled_qty / order_qty × 100.
    avg_fill_price   : VWAP of all fills.
    vwap_ref         : Bar VWAP at order time — the reference benchmark price.
    slippage_bps     : (avg_fill − vwap_ref) / vwap_ref × 10 000.
                       Positive = paid more than benchmark (bad for BUY).
    fees_paid        : Dollar fees charged by venues.
    total_notional   : avg_fill_price × filled_qty.
    total_cost       : total_notional + fees_paid  (all-in dollar cost).
    cost_per_share   : total_cost / filled_qty  (NaN when unfilled).
    implementation_shortfall_bps :
        (total_cost / filled_qty − vwap_ref) / vwap_ref × 10 000.
        Combines slippage AND fees into a single quality number.
    execution_time_ms: Cumulative simulated latency across all child orders.
    num_venues_used  : How many distinct exchanges received a fill.
    venues_used      : Comma-separated venue names that filled.
    """

    window_idx: int
    timestamp: str
    strategy: str

    order_qty: float
    filled_qty: float
    unfilled_qty: float
    fill_rate_pct: float

    avg_fill_price: float
    vwap_ref: float
    slippage_bps: float
    fees_paid: float

    total_notional: float
    total_cost: float
    cost_per_share: float
    implementation_shortfall_bps: float

    execution_time_ms: int
    num_venues_used: int
    venues_used: str


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Rolls through AAPL 1-minute bars window-by-window, running all four
    routing strategies on each window and recording a BacktestRecord for
    every (window × strategy) pair.

    Parameters
    ----------
    csv_path    : Path to ``AAPL_1min_2024-2026.csv``.
    window_size : Bars per execution window; also the TWAP/VWAP slice count.
                  Default 5.
    order_qty   : Shares per synthetic parent order.  5 000 stresses venue
                  depth enough to produce visible fill-rate differences.
    side        : BUY (default) or SELL.
    max_windows : Cap on windows processed.  200 windows × 5 bars = 1 000 bars
                  of real AAPL data; runs in ~10 seconds.
    log_every   : Print a progress line every N windows.  Default 50.
    """

    def __init__(
        self,
        csv_path: str,
        window_size: int = 5,
        order_qty: float = 5_000.0,
        side: OrderSide = OrderSide.BUY,
        max_windows: int = 200,
        log_every: int = 50,
    ) -> None:
        self.csv_path    = csv_path
        self.window_size = window_size
        self.order_qty   = order_qty
        self.side        = side
        self.max_windows = max_windows
        self.log_every   = log_every

        self._strategies = [
            BestPriceStrategy(),
            TWAPStrategy(num_slices=window_size),
            VWAPStrategy(num_slices=window_size),
            SmartStrategy(),
        ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full backtest and return a ``pandas.DataFrame`` of records.

        Each row is one (window, strategy) observation.  The DataFrame has
        ``len(windows) × 4`` rows and every column in ``BacktestRecord``.
        """
        log.info("Loading AAPL price feed from %s", self.csv_path)
        feed = PriceFeed(self.csv_path)
        all_bars = list(feed)

        total_windows = min(self.max_windows, len(all_bars) // self.window_size)
        log.info(
            "Backtest: %d windows × %d bars × %d strategies = %d simulations",
            total_windows, self.window_size, len(self._strategies),
            total_windows * len(self._strategies),
        )

        records: List[BacktestRecord] = []

        for w in range(total_windows):
            start  = w * self.window_size
            window = all_bars[start : start + self.window_size]
            if len(window) < self.window_size:
                break

            ref_bar = window[0]   # VWAP / timestamp reference

            for strategy in self._strategies:
                # Each strategy gets its own fresh, identically-seeded venues.
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
                records.append(self._make_record(w, ref_bar, result))

            if self.log_every > 0 and (w + 1) % self.log_every == 0:
                log.info("  %d / %d windows complete", w + 1, total_windows)

        log.info("Backtest finished — %d records generated", len(records))
        return pd.DataFrame([asdict(r) for r in records])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _make_record(
        self,
        window_idx: int,
        ref_bar: Bar,
        result: RouterResult,
    ) -> BacktestRecord:
        filled  = result.total_filled
        unfilled = self.order_qty - filled

        total_notional = result.avg_fill_price * filled if filled > 0 else 0.0
        total_cost     = total_notional + result.total_fees

        if filled > 0:
            cost_per_share = total_cost / filled
            is_bps = (cost_per_share - ref_bar.vwap) / ref_bar.vwap * 10_000.0
        else:
            cost_per_share = float("nan")
            is_bps         = float("nan")

        venues_list = result.venues_used   # already filtered to filled only

        return BacktestRecord(
            window_idx   = window_idx,
            timestamp    = ref_bar.timestamp.isoformat(),
            strategy     = result.strategy_name,

            order_qty    = self.order_qty,
            filled_qty   = round(filled, 2),
            unfilled_qty = round(unfilled, 2),
            fill_rate_pct= round(result.fill_rate_pct, 4),

            avg_fill_price = round(result.avg_fill_price, 6),
            vwap_ref       = round(ref_bar.vwap, 6),
            slippage_bps   = round(result.total_slippage_bps, 4),
            fees_paid      = round(result.total_fees, 6),

            total_notional = round(total_notional, 4),
            total_cost     = round(total_cost, 4),
            cost_per_share = round(cost_per_share, 6) if filled > 0 else float("nan"),
            implementation_shortfall_bps = round(is_bps, 4) if filled > 0 else float("nan"),

            execution_time_ms = result.execution_time_ms,
            num_venues_used   = len(venues_list),
            venues_used       = ",".join(venues_list) if venues_list else "",
        )
