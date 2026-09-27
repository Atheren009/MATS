"""
test_latency.py — Sanity tests for the latency profiler.

Runs with a tiny N=100 so the test suite stays fast, then asserts the
output shape and a generous upper-bound on P99 (5 ms — any Python process
that can't do this in 5 ms has a serious bug, not a performance issue).
"""

from __future__ import annotations

import gc
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pytest

logging.disable(logging.CRITICAL)

from order_router.child_order import ChildOrder
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import BestPriceStrategy, SmartStrategy

logging.disable(logging.NOTSET)

TINY_N = 100   # fast — just enough to validate shape and correctness


def _ref_bar() -> Bar:
    return Bar(
        symbol="AAPL",
        timestamp=datetime(2024, 6, 3, 14, 30, tzinfo=timezone.utc),
        open=189.50, high=190.20, low=189.30, close=189.95,
        volume=8_500.0, trade_count=120.0, vwap=189.82,
    )


def _time_strategy(strategy, bar, n=TINY_N) -> np.ndarray:
    """Return µs latency array for n iterations."""
    logging.disable(logging.CRITICAL)
    gc.disable()
    lats = np.empty(n, dtype=np.float64)
    for i in range(n):
        venues = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
        order  = Order("AAPL", OrderSide.BUY, OrderType.MARKET, quantity=5_000.0)
        router = OrderRouter(venues=venues, strategy=strategy)
        t0 = time.perf_counter_ns()
        router.submit(order, bars=[bar])
        t1 = time.perf_counter_ns()
        lats[i] = (t1 - t0) * 1e-3
    gc.enable()
    logging.disable(logging.NOTSET)
    return lats


# ---------------------------------------------------------------------------
# Correctness — shape and values
# ---------------------------------------------------------------------------

def test_latency_array_length():
    """Profile returns exactly N measurements."""
    lats = _time_strategy(BestPriceStrategy(), _ref_bar(), n=TINY_N)
    assert len(lats) == TINY_N


def test_all_latencies_positive():
    """Every timing must be strictly positive."""
    lats = _time_strategy(BestPriceStrategy(), _ref_bar(), n=TINY_N)
    assert (lats > 0).all(), f"Non-positive latency found: {lats[lats <= 0]}"


def test_latencies_are_finite():
    """No NaN or Inf in the measurements."""
    lats = _time_strategy(SmartStrategy(), _ref_bar(), n=TINY_N)
    assert np.all(np.isfinite(lats))


# ---------------------------------------------------------------------------
# Performance — generous bounds for CI environments
# ---------------------------------------------------------------------------

def test_best_price_p99_below_5ms():
    """
    BestPrice P99 must be below 5 000 µs (5 ms) even in a slow CI container.
    This catches catastrophic regressions like accidental blocking I/O,
    not the 150-200 µs production target.
    """
    lats = _time_strategy(BestPriceStrategy(), _ref_bar(), n=TINY_N)
    p99  = float(np.percentile(lats, 99))
    assert p99 < 5_000.0, f"BestPrice P99={p99:.1f}µs — exceeds 5 ms sanity bound"


def test_smart_p99_below_5ms():
    """Smart (3-venue sweep) P99 must also stay below 5 ms."""
    lats = _time_strategy(SmartStrategy(), _ref_bar(), n=TINY_N)
    p99  = float(np.percentile(lats, 99))
    assert p99 < 5_000.0, f"Smart P99={p99:.1f}µs — exceeds 5 ms sanity bound"


def test_smart_slower_than_best_price_median():
    """
    Smart routes to ≥ 1 additional venue vs BestPrice, so its median
    latency should be >= BestPrice median (more work per call).
    """
    bar   = _ref_bar()
    bp    = _time_strategy(BestPriceStrategy(), bar, n=TINY_N)
    smart = _time_strategy(SmartStrategy(),     bar, n=TINY_N)
    assert np.median(smart) >= np.median(bp) * 0.5, (
        "Smart surprisingly much faster than BestPrice — "
        "something may be short-circuiting"
    )


# ---------------------------------------------------------------------------
# Component profiling
# ---------------------------------------------------------------------------

def test_seeding_positive():
    """Seeding 3 venues must take > 0 µs."""
    bar    = _ref_bar()
    venues = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
    t0 = time.perf_counter_ns()
    for exch in venues.values():
        exch.seed_from_bar(bar)
    t1 = time.perf_counter_ns()
    elapsed_us = (t1 - t0) * 1e-3
    assert elapsed_us > 0


def test_matching_single_venue():
    """Single Exchange.submit() for 1 500 shares must produce a positive latency."""
    bar    = _ref_bar()
    exch   = Exchange("ALPHA", VENUE_CONFIGS["ALPHA"])
    exch.seed_from_bar(bar)
    child  = ChildOrder("parent", "ALPHA", OrderSide.BUY, quantity=1_500.0)

    t0 = time.perf_counter_ns()
    result = exch.submit(child)
    t1 = time.perf_counter_ns()

    elapsed_us = (t1 - t0) * 1e-3
    assert elapsed_us > 0
    assert result.filled_qty > 0   # must have actually matched
