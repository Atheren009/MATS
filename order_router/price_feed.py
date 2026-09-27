"""
PriceFeed — streams AAPL 1-minute bars from the local CSV file.

The feed acts as a realistic market price source for testing and simulation.
Each call to `next_bar()` advances one minute and returns an OHLCV bar.

Usage
-----
>>> feed = PriceFeed("AAPL_1min_2024-2026.csv")
>>> bar = feed.next_bar()
>>> bar.mid_price   # (high + low) / 2 — used to seed order prices in tests
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class Bar:
    """A single 1-minute OHLCV bar."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: float
    vwap: float

    @property
    def mid_price(self) -> float:
        """Midpoint of high and low — a simple fair-value estimate."""
        return round((self.high + self.low) / 2, 4)

    @property
    def typical_price(self) -> float:
        """(High + Low + Close) / 3 — standard technical analysis reference price."""
        return round((self.high + self.low + self.close) / 3, 4)


class PriceFeed:
    """
    Iterates over AAPL 1-minute bars from a CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file produced by codecheck.py.
    symbol_filter : str, optional
        If set, only rows for this symbol are yielded.
    """

    def __init__(
        self,
        csv_path: str | Path,
        symbol_filter: str = "AAPL",
    ) -> None:
        self._path = Path(csv_path)
        self._symbol_filter = symbol_filter
        self._bars: List[Bar] = []
        self._index: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterator[Bar]:
        return iter(self._bars)

    def reset(self) -> None:
        """Rewind the feed to the first bar."""
        self._index = 0

    def next_bar(self) -> Optional[Bar]:
        """
        Advance one bar and return it, or None if the feed is exhausted.
        """
        if self._index >= len(self._bars):
            return None
        bar = self._bars[self._index]
        self._index += 1
        return bar

    def peek(self) -> Optional[Bar]:
        """Return the next bar without advancing the cursor."""
        if self._index >= len(self._bars):
            return None
        return self._bars[self._index]

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._bars)

    def bars_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Bar]:
        """Return all bars whose timestamp falls within [start, end]."""
        # Make both timezone-aware or both naive for comparison
        result = []
        for b in self._bars:
            ts = b.timestamp
            # Normalise: strip tz info if start/end are naive
            if start.tzinfo is None:
                ts = ts.replace(tzinfo=None)
            if start <= ts <= end:
                result.append(b)
        return result

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Parse the CSV and populate self._bars."""
        with open(self._path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if self._symbol_filter and row["symbol"] != self._symbol_filter:
                    continue
                self._bars.append(
                    Bar(
                        symbol=row["symbol"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        trade_count=float(row["trade_count"]),
                        vwap=float(row["vwap"]),
                    )
                )
