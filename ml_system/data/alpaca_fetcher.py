"""
AlpacaFetcher — fetch historical bars from Alpaca (equities) or yfinance (XAUUSD).

Checks local CSV coverage first and only hits the API for missing date ranges.
Saves fetched data back to CSV so subsequent runs skip the download.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    CSV_PATHS,
    DATA_DIR,
    EQUITY_SYMBOLS,
    ALPACA_EXTRA_SYMBOLS,
    GOLD_SYMBOL,
    GOLD_FUTURES_SYMBOL,
    YFINANCE_TICKERS,
)

log = logging.getLogger(__name__)


class AlpacaFetcher:
    """
    Fetch historical bar data from Alpaca or yfinance.

    For equities: uses the Alpaca StockHistoricalDataClient.
    For XAUUSD:   uses yfinance with the GC=F (gold futures) ticker.

    All data is returned as a pandas DataFrame with a UTC DatetimeIndex
    and columns: open, high, low, close, volume.
    """

    def __init__(self) -> None:
        self._alpaca_client = None
        self._init_alpaca()

    def _init_alpaca(self) -> None:
        """Lazy-init the Alpaca client."""
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            self._alpaca_client = StockHistoricalDataClient(
                ALPACA_API_KEY, ALPACA_SECRET_KEY
            )
            log.info("Alpaca client initialised (paper account)")
        except ImportError:
            log.warning(
                "alpaca-py not installed — Alpaca data fetch disabled. "
                "Install with: pip install alpaca-py"
            )
        except Exception as exc:
            log.warning("Alpaca client init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe_minutes: int = 1,
    ) -> pd.DataFrame:
        """
        Fetch bars for *symbol* between *start* and *end*.

        Defaults: start = 2 years ago, end = now.
        Returns DataFrame with UTC DatetimeIndex.
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(days=730)

        log.info(
            "Fetching %s  %s → %s  (%dmin bars)",
            symbol,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
            timeframe_minutes,
        )

        if symbol in [GOLD_SYMBOL, GOLD_FUTURES_SYMBOL]:
            df = self._fetch_yfinance(symbol, start, end, timeframe_minutes)
        else:
            df = self._fetch_alpaca(symbol, start, end, timeframe_minutes)

        # Persist to CSV for next run
        self._save_csv(symbol, df, timeframe_minutes)
        return df

    def fetch_if_missing(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        timeframe_minutes: int = 1,
    ) -> pd.DataFrame:
        """
        Load from CSV if it covers [start, end]; otherwise fetch the gap.
        """
        csv_path = CSV_PATHS.get(symbol)
        if csv_path and Path(csv_path).exists():
            existing = pd.read_csv(csv_path)
            if "symbol" in existing.columns:
                existing = existing[existing["symbol"] == symbol]
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            existing = existing.set_index("timestamp").sort_index()

            existing_start = existing.index[0]
            existing_end = existing.index[-1]

            if start is None:
                start = existing_start
            if end is None:
                end = datetime.now(timezone.utc)

            # If existing data fully covers the requested range, use it
            if existing_start <= start and existing_end >= end - timedelta(hours=1):
                log.info(
                    "  Local CSV covers %s → %s — no fetch needed",
                    existing_start.strftime("%Y-%m-%d"),
                    existing_end.strftime("%Y-%m-%d"),
                )
                return existing

            log.info(
                "  Local CSV partial: %s → %s.  Fetching remainder.",
                existing_start.strftime("%Y-%m-%d"),
                existing_end.strftime("%Y-%m-%d"),
            )

        return self.fetch(symbol, start, end, timeframe_minutes)

    # ------------------------------------------------------------------
    # Alpaca (equities)
    # ------------------------------------------------------------------

    def _fetch_alpaca(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe_minutes: int,
    ) -> pd.DataFrame:
        """Fetch stock bars from Alpaca."""
        if self._alpaca_client is None:
            raise RuntimeError("Alpaca client not available")

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        if timeframe_minutes == 1:
            tf = TimeFrame.Minute
        elif timeframe_minutes == 5:
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif timeframe_minutes == 15:
            tf = TimeFrame(15, TimeFrameUnit.Minute)
        elif timeframe_minutes == 60:
            tf = TimeFrame.Hour
        else:
            tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=tf,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )

        bars = self._alpaca_client.get_stock_bars(request)
        df = bars.df

        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "timestamp"

        # Standardise column names
        col_map = {
            "trade_count": "trade_count",
            "vwap": "vwap",
        }
        for old, new in col_map.items():
            if old in df.columns and old != new:
                df = df.rename(columns={old: new})

        log.info("  Alpaca returned %d bars for %s", len(df), symbol)
        return df

    # ------------------------------------------------------------------
    # yfinance (XAUUSD via gold futures)
    # ------------------------------------------------------------------

    def _fetch_yfinance(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe_minutes: int,
    ) -> pd.DataFrame:
        """
        Fetch gold futures (GC=F) from yfinance.

        yfinance intraday limits:
            1min  → last 7 days
            15min → last 60 days
            1h    → last 730 days
            1d    → unlimited

        Strategy: fetch at the finest resolution available, then
        the caller can resample down.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance not installed.  Install with: pip install yfinance"
            )

        ticker = YFINANCE_TICKERS.get(symbol, symbol)
        days_back = (end - start).days

        # Choose interval based on what yfinance allows
        if timeframe_minutes <= 1 and days_back <= 7:
            interval = "1m"
        elif timeframe_minutes <= 5 and days_back <= 60:
            interval = "5m"
        elif timeframe_minutes <= 15 and days_back <= 60:
            interval = "15m"
        elif timeframe_minutes <= 60 and days_back <= 730:
            interval = "1h"
        else:
            # Fall back: fetch 1h for max history
            interval = "1h"
            
            # Clip start date to max 729 days ago from TODAY for yfinance 1h limit
            max_start = datetime.now(timezone.utc) - timedelta(days=729)
            # Remove timezone awareness from max_start if start is naive, or add to max_start
            if start.tzinfo is None:
                max_start = max_start.replace(tzinfo=None)
            if start < max_start:
                log.warning(
                    "  yfinance: requested start %s is > 730 days ago. Clipping to %s",
                    start.strftime("%Y-%m-%d"), max_start.strftime("%Y-%m-%d")
                )
                start = max_start
                
            log.warning(
                "  yfinance: requested %d days at %dmin — "
                "falling back to 1h (max 730 days intraday)",
                days_back, timeframe_minutes,
            )

        log.info(
            "  yfinance: fetching %s (%s) interval=%s",
            symbol, ticker, interval,
        )

        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
        )

        if data.empty:
            raise RuntimeError(
                f"yfinance returned no data for {ticker} "
                f"({start.date()} → {end.date()}, interval={interval})"
            )

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Standardise
        df = data.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
            "Adj Close": "adj_close",
        })

        keep = ["open", "high", "low", "close", "volume"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # Ensure UTC
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "timestamp"

        # Add trade_count and vwap stubs (not available from yfinance)
        df["trade_count"] = np.nan
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0

        log.info("  yfinance returned %d bars for %s", len(df), symbol)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_csv(
        self, symbol: str, df: pd.DataFrame, timeframe_minutes: int
    ) -> None:
        """Save fetched data to CSV in the Price Feed directory."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if timeframe_minutes == 1:
            tf_label = "1min"
        elif timeframe_minutes == 15:
            tf_label = "15min"
        elif timeframe_minutes == 60:
            tf_label = "1h"
        else:
            tf_label = f"{timeframe_minutes}min"

        start_year = df.index[0].year
        end_year = df.index[-1].year
        filename = f"{symbol}_{tf_label}_{start_year}-{end_year}.csv"
        path = DATA_DIR / filename

        df_out = df.copy()
        df_out.insert(0, "symbol", symbol)
        df_out.to_csv(path)

        log.info("  Saved %d bars → %s", len(df), path)

        # Register in CSV_PATHS so loader can find it
        CSV_PATHS[symbol] = path
