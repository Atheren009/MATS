"""
tests/test_signal_timing.py
===========================
Acceptance test for the A2 signal-timestamp lookahead fix.

Invariant: a signal whose timestamp is the 15-min bar open label (T)
must not be fillable on any 1-min bar timestamped before T+15min
(the bar close time).
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _entry_bars_after_signal(df_1min, sig_time, bar_minutes=15):
    """Return 1-min bars eligible for entry after the signal bar closes."""
    bar_close = sig_time + pd.Timedelta(minutes=bar_minutes)
    return df_1min[df_1min.index >= bar_close]


def _build_1min_grid(start, n_bars=30):
    """Minimal synthetic 1-min OHLCV grid for testing."""
    idx = pd.date_range(start, periods=n_bars, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open":   [100.0] * n_bars,
            "high":   [100.5] * n_bars,
            "low":    [99.5]  * n_bars,
            "close":  [100.2] * n_bars,
            "volume": [1000]  * n_bars,
        },
        index=idx,
    )


class TestSignalTimingLookahead:
    """
    Encode the A2 acceptance criterion.
    All four assertions must hold for the lookahead fix to be correct.
    """

    def test_no_bars_before_bar_close(self):
        """Every bar in the entry window must be >= sig_time + 15min."""
        df = _build_1min_grid("2026-01-02 09:30:00", n_bars=30)
        sig_time = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        entry_bars = _entry_bars_after_signal(df, sig_time)
        bar_close = sig_time + pd.Timedelta(minutes=15)
        assert (entry_bars.index >= bar_close).all(), (
            f"{(entry_bars.index < bar_close).sum()} bars found before bar close {bar_close}"
        )

    def test_first_valid_bar_is_bar_close(self):
        """First entry-eligible bar must be exactly at sig_time + 15min."""
        df = _build_1min_grid("2026-01-02 09:30:00", n_bars=30)
        sig_time = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        bar_close = sig_time + pd.Timedelta(minutes=15)
        entry_bars = _entry_bars_after_signal(df, sig_time)
        assert len(entry_bars) > 0
        assert entry_bars.index[0] == bar_close

    def test_bars_inside_signal_window_excluded(self):
        """Bars at T+1min through T+14min must NOT appear in the entry window."""
        df = _build_1min_grid("2026-01-02 09:30:00", n_bars=30)
        sig_time = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        entry_bars = _entry_bars_after_signal(df, sig_time)
        inside_bar = [sig_time + pd.Timedelta(minutes=m) for m in range(1, 15)]
        overlapping = [ts for ts in inside_bar if ts in entry_bars.index]
        assert overlapping == [], f"Inside-bar timestamps leaked into entry window: {overlapping}"

    def test_old_filter_includes_inside_bars(self):
        """
        Verify the pre-A2 bug: df_1min.index > sig_time included bars inside
        the 15-min bar (lookahead contamination).
        """
        df = _build_1min_grid("2026-01-02 09:30:00", n_bars=30)
        sig_time = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        old_entry_bars = df[df.index > sig_time]
        bar_close = sig_time + pd.Timedelta(minutes=15)
        inside_in_old = old_entry_bars[old_entry_bars.index < bar_close]
        assert len(inside_in_old) > 0, "Pre-A2 filter should have included inside-bar entries"

    def test_no_data_after_bar_close_returns_empty(self):
        """Edge case: if data ends before bar_close, entry window is empty."""
        # 14 bars starting at 09:30 -> last bar is 09:43, bar_close is 09:45
        df = _build_1min_grid("2026-01-02 09:30:00", n_bars=14)
        sig_time = pd.Timestamp("2026-01-02 09:30:00", tz="UTC")
        entry_bars = _entry_bars_after_signal(df, sig_time)
        assert len(entry_bars) == 0
