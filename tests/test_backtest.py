"""
test_backtest.py — Sanity tests for the BacktestEngine.

These tests run on a very small number of windows (5) so the test suite
stays fast while verifying correctness of the engine's output shape,
column completeness, and the core thesis: Smart fills more than BestPrice.
"""

from __future__ import annotations

import os
import math

import pandas as pd
import pytest

from order_router.backtest import BacktestEngine, STRATEGY_ORDER
from order_router.models import OrderSide

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "AAPL_1min_2024-2026.csv")
TINY_WINDOWS = 5   # keep tests fast


@pytest.fixture(scope="module")
def backtest_df() -> pd.DataFrame:
    """Run a small backtest once per module and share the result."""
    engine = BacktestEngine(
        csv_path    = CSV_PATH,
        window_size = 5,
        order_qty   = 5_000.0,
        side        = OrderSide.BUY,
        max_windows = TINY_WINDOWS,
        log_every   = 0,   # suppress progress logs in tests
    )
    return engine.run()


# ---------------------------------------------------------------------------
# Shape & schema
# ---------------------------------------------------------------------------

def test_backtest_row_count(backtest_df):
    """One row per (window × strategy) — 5 windows × 4 strategies = 20 rows."""
    assert len(backtest_df) == TINY_WINDOWS * len(STRATEGY_ORDER)


def test_backtest_has_all_strategies(backtest_df):
    """Every strategy must appear in the results."""
    found = set(backtest_df["strategy"].unique())
    assert set(STRATEGY_ORDER) == found


def test_backtest_required_columns(backtest_df):
    """All expected CSV columns are present."""
    required = {
        "window_idx", "timestamp", "strategy",
        "order_qty", "filled_qty", "unfilled_qty", "fill_rate_pct",
        "avg_fill_price", "vwap_ref", "slippage_bps", "fees_paid",
        "total_notional", "total_cost", "cost_per_share",
        "implementation_shortfall_bps",
        "execution_time_ms", "num_venues_used", "venues_used",
    }
    assert required.issubset(set(backtest_df.columns))


# ---------------------------------------------------------------------------
# Value sanity
# ---------------------------------------------------------------------------

def test_fill_rate_in_range(backtest_df):
    """Fill rate must be between 0 and 100 for every row."""
    assert backtest_df["fill_rate_pct"].between(0, 100).all()


def test_unfilled_plus_filled_equals_order_qty(backtest_df):
    """filled_qty + unfilled_qty must equal order_qty for every row."""
    diff = (backtest_df["filled_qty"] + backtest_df["unfilled_qty"] - backtest_df["order_qty"]).abs()
    assert (diff < 1e-3).all(), f"Max diff: {diff.max()}"


def test_fees_positive_when_filled(backtest_df):
    """Any row with filled_qty > 0 must have fees_paid > 0."""
    filled_rows = backtest_df[backtest_df["filled_qty"] > 0]
    assert (filled_rows["fees_paid"] > 0).all()


def test_cost_per_share_nan_when_unfilled(backtest_df):
    """cost_per_share must be NaN when filled_qty == 0."""
    unfilled = backtest_df[backtest_df["filled_qty"] == 0]
    if len(unfilled):
        assert unfilled["cost_per_share"].isna().all()


# ---------------------------------------------------------------------------
# The thesis: Smart > BestPrice
# ---------------------------------------------------------------------------

def test_smart_fill_rate_ge_best_price(backtest_df):
    """
    Smart routing must achieve at least as high an average fill rate as
    BestPrice across all windows.  This is the core claim of Phase 4.
    """
    agg = backtest_df.groupby("strategy")["fill_rate_pct"].mean()
    assert agg["Smart"] >= agg["BestPrice"], (
        f"Smart ({agg['Smart']:.2f}%) should be >= BestPrice ({agg['BestPrice']:.2f}%)"
    )


def test_smart_lower_cumulative_unfilled(backtest_df):
    """Smart must leave fewer total unfilled shares than BestPrice."""
    totals = backtest_df.groupby("strategy")["unfilled_qty"].sum()
    assert totals["Smart"] <= totals["BestPrice"], (
        f"Smart unfilled ({totals['Smart']:,.0f}) should be <= BestPrice ({totals['BestPrice']:,.0f})"
    )
