"""
DataLoader — Load 1-minute CSVs and resample to any timeframe.

Computes all technical features required by the HMM and baseline signal
generator in a single pass.  The model never sees raw 1-minute bars; this
module is the only place timeframe conversion happens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from ..config import (
    CSV_PATHS,
    DEFAULT_TIMEFRAME,
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    EQUITY_SYMBOLS,
    ALPACA_EXTRA_SYMBOLS,
    GOLD_SYMBOL,
    GOLD_FUTURES_SYMBOL,
    FORWARD_N_BARS,
)

log = logging.getLogger(__name__)


class DataLoader:
    """
    Load and resample OHLCV data from local 1-minute CSVs.

    Parameters
    ----------
    symbol : str
        Ticker symbol (AAPL, MSFT, SPY, or XAUUSD).
    csv_path : Path, optional
        Override path to CSV.  Falls back to config CSV_PATHS.
    """

    def __init__(self, symbol: str, csv_path: Optional[Path] = None) -> None:
        self.symbol = symbol
        self.csv_path = csv_path or CSV_PATHS.get(symbol)
        self._df_1min: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load 1-minute bars from CSV into a tz-aware DatetimeIndex DataFrame."""
        if self.csv_path is None or not Path(self.csv_path).exists():
            raise FileNotFoundError(
                f"No local CSV for {self.symbol}.  "
                f"Use AlpacaFetcher to download first."
            )

        log.info("Loading %s from %s", self.symbol, self.csv_path)
        df = pd.read_csv(self.csv_path)

        # Filter to the requested symbol (CSVs may contain multiple)
        if "symbol" in df.columns:
            df = df[df["symbol"] == self.symbol].copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()

        # Keep only numeric OHLCV columns we need
        keep = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].astype(float)

        self._df_1min = df
        log.info(
            "  Loaded %s: %d bars, %s → %s",
            self.symbol, len(df),
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
        )
        return df

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Inject a pre-loaded DataFrame (used by AlpacaFetcher)."""
        self._df_1min = df

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(self, tf: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
        """
        Resample 1-minute bars to the target timeframe.

        Correct OHLCV aggregation: open=first, high=max, low=min,
        close=last, volume=sum.
        """
        if self._df_1min is None:
            self.load()

        agg_rules = {
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last",
            "volume": "sum",
        }
        # Optional columns
        if "trade_count" in self._df_1min.columns:
            agg_rules["trade_count"] = "sum"
        if "vwap" in self._df_1min.columns:
            agg_rules["vwap"] = "last"

        cols = [c for c in agg_rules if c in self._df_1min.columns]
        # label='right', closed='right': the bar whose timestamp is T contains
        # 1-min data from (T - tf, T].  A signal generated at bar T is therefore
        # only actionable on bars with index > T — perfectly causal, no lookahead.
        df = self._df_1min[cols].resample(tf, label="right", closed="right").agg(
            {c: agg_rules[c] for c in cols}
        )
        df = df.dropna(subset=["open"])  # drop empty bars
        log.info(
            "  Resampled %s to %s: %d bars", self.symbol, tf, len(df)
        )
        return df

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features needed by HMM + baseline signals.

        Returns a copy with new columns appended.  Drops rows with NaN
        from warmup periods.
        """
        out = df.copy()

        # --- Log return ---
        out["log_return"] = np.log(out["close"] / out["close"].shift(1))

        # --- True Range & ATR(14) ---
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr14"] = tr.rolling(ATR_PERIOD).mean()
        out["atr14_norm"] = out["atr14"] / out["close"]

        # --- Realised volatility (20-bar rolling std of log returns) ---
        out["realised_vol_20"] = out["log_return"].rolling(20).std()

        # --- High-Low range / ATR ratio (expansion vs compression) ---
        out["hl_range_atr"] = (out["high"] - out["low"]) / out["atr14"]

        # --- Body-to-wick ratio (directional conviction) ---
        body = (out["close"] - out["open"]).abs()
        wick = out["high"] - out["low"]
        out["body_ratio"] = body / wick.replace(0, np.nan)

        # --- Relative volume ---
        vol_sma = out["volume"].rolling(20).mean()
        out["rel_volume"] = out["volume"] / vol_sma

        # --- EMA 9 / 21 (for baseline crossover signals) ---
        out["ema_fast"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
        out["ema_slow"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()

        # --- EMA 200 (primary trend direction — bull or bear bias) ---
        # On 15-min bars, EMA200 spans ~50 hours of price action (~3 trading days)
        # This is our "trend compass": above = bullish, below = bearish
        out["ema_trend"] = out["close"].ewm(span=200, adjust=False).mean()
        out["trend_bias"] = (out["close"] > out["ema_trend"]).astype(int)  # 1=bullish, 0=bearish

        # --- RSI(14) ---
        delta = out["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / avg_loss
        out["rsi14"] = 100.0 - (100.0 / (1.0 + rs))

        # --- Session classification ---
        out = self._add_session(out)

        # Drop warmup NaN rows
        out = out.dropna()
        log.info("  Features computed: %d rows, %d columns", len(out), len(out.columns))
        return out

    # ------------------------------------------------------------------
    # Session tagging
    # ------------------------------------------------------------------

    def _add_session(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tag each bar with a session label.

        Equities (US Eastern):
            pre_market:  04:00–09:30
            regular:     09:30–16:00
            after_hours: 16:00–20:00

        XAUUSD (UTC):
            asian:   00:00–08:00
            london:  08:00–13:00  (overlap with NY after 13:00)
            ny:      13:00–22:00
            low_liq: 22:00–00:00
        """
        idx = df.index

        if self.symbol in [GOLD_SYMBOL, GOLD_FUTURES_SYMBOL]:
            utc_hour = idx.hour
            conditions = [
                (utc_hour >= 0) & (utc_hour < 8),
                (utc_hour >= 8) & (utc_hour < 13),
                (utc_hour >= 13) & (utc_hour < 22),
                (utc_hour >= 22),
            ]
            choices = ["asian", "london", "ny", "low_liq"]
            df["session"] = np.select(conditions, choices, default="low_liq")

        else:
            # All equity symbols default to US/Eastern market sessions
            eastern = idx.tz_convert("US/Eastern")
            hour_min = eastern.hour * 100 + eastern.minute

            conditions = [
                (hour_min >= 400) & (hour_min < 930),
                (hour_min >= 930) & (hour_min < 1600),
                (hour_min >= 1600) & (hour_min < 2000),
            ]
            choices = ["pre_market", "regular", "after_hours"]
            df["session"] = np.select(conditions, choices, default="closed")

        return df

    # ------------------------------------------------------------------
    # LSTM Features (Phase 1)
    # ------------------------------------------------------------------

    def compute_lstm_features(self, df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Compute additional features for the LSTM filter:
        - Distance to rolling 24h High/Low (proxy for PDH/PDL)
        - Regime one-hot
        - Session one-hot
        - Target variable (forward return sign)
        
        Requires `df` to have already been passed through `compute_features()`
        and `regimes` to be the HMM predictions.
        """
        out = df.copy()
        
        # 1. Structural Proxies (Rolling 24h High/Low at 15m = 96 bars)
        # Using 96 bars as a robust approximation for PDH/PDL without looking ahead
        rolling_high = out["high"].rolling(96).max()
        rolling_low = out["low"].rolling(96).min()
        
        out["dist_pdh"] = (rolling_high - out["close"]) / out["atr14"]
        out["dist_pdl"] = (out["close"] - rolling_low) / out["atr14"]
        
        # 2. One-Hot Encoded Regime
        out["regime"] = regimes
        regime_dummies = pd.get_dummies(out["regime"], prefix="regime")
        # Ensure all 4 HMM regimes exist
        for r in ["trending", "ranging", "high_vol_breakout", "low_liquidity"]:
            col = f"regime_{r}"
            if col not in regime_dummies.columns:
                regime_dummies[col] = 0.0
        out = pd.concat([out, regime_dummies.astype(float)], axis=1)
        
        # 3. One-Hot Encoded Session
        session_dummies = pd.get_dummies(out["session"], prefix="session")
        out = pd.concat([out, session_dummies.astype(float)], axis=1)
        
        # 4. Target Variable (N-bar forward return sign)
        # 1 if positive return, 0 if negative/flat
        forward_return = np.log(out["close"].shift(-FORWARD_N_BARS) / out["close"])
        out["target"] = (forward_return > 0).astype(float)
        
        # Drop rows with NaN (from rolling calculations or the shift at the end)
        out = out.dropna()
        log.info("  LSTM features computed: %d rows, %d columns", len(out), len(out.columns))
        return out


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for LSTM sequences.
    Creates sliding windows of length `seq_len` over the feature matrix.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
        
    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len + 1)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Sequence of length seq_len
        x_seq = self.X[idx : idx + self.seq_len]
        # Target corresponding to the LAST bar in the sequence
        y_val = self.y[idx + self.seq_len - 1]
        return x_seq, y_val
