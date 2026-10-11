"""
BaselineSignalGenerator — Phase 0 rule-based signals filtered by HMM regime.

This is the *permanent anchor* baseline.  The LSTM (Phase 1) must beat this
on DSR + SPA before it can be promoted.  This module is never retired.

Signal rules (v2 — improved entry quality + RR):
    trending regime     → EMA(9)/EMA(21) crossover, confirmed by volume + body conviction
    ranging regime      → RSI(14) mean-reversion (< 25 long, > 75 short), 2-bar extreme confirmation
    high_vol_breakout   → NO TRADE (too noisy for rule-based)
    low_liquidity       → NO TRADE (pre-market / Asian session)

Quality filters applied to all signals:
    - ATR must be > MIN_ATR_PCT of price (skip dead/choppy bars)
    - Equities: only trade 9:45–15:15 ET (avoid open/close spikes)
    - EMA crossover: requires rel_volume > VOL_CONFIRM and body_ratio > BODY_CONFIRM
    - RSI reversion: RSI must have been in extreme zone for >= 2 bars
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import LSTM_PROB_THRESH

from ..config import (
    ATR_STOP_MULT,
    ATR_TP_MULT,
    EMA_FAST,
    EMA_SLOW,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    EQUITY_SYMBOLS,
    ALPACA_EXTRA_SYMBOLS,
    GOLD_SYMBOL,
    GOLD_FUTURES_SYMBOL,
)
from ..models.hmm_regime import (
    TRENDING,
    RANGING,
    NO_TRADE_REGIMES,
)

log = logging.getLogger(__name__)

# ----- Quality filter constants -----
MIN_ATR_PCT   = 0.001   # ATR must be > 0.1% of price (skip dead bars)
VOL_CONFIRM   = 1.1     # rel_volume must be above average for crossovers
BODY_CONFIRM  = 0.50    # body_ratio for EMA crossover conviction (raised from 0.45)
RSI_MIN_BARS  = 2       # RSI must be in extreme zone for this many bars before reversion


@dataclass(frozen=True)
class Signal:
    """A single trade signal."""

    symbol: str
    timestamp: str           # ISO format
    direction: str           # "long" or "short"
    strength: float          # 0.0–1.0 confidence proxy
    regime: str              # HMM regime label at signal time
    session: str             # Session tag at signal time
    entry_price: float       # Close at signal bar
    stop_price: float        # ATR-based stop loss
    tp_price: float          # ATR-based take profit
    atr: float               # ATR(14) at signal bar
    rule: str                # Which rule generated this signal
    lstm_prob: Optional[float] = None  # LSTM confidence (if used)

    def to_dict(self) -> dict:
        return asdict(self)


class BaselineSignalGenerator:
    """
    Generate regime-filtered signals using EMA crossover and RSI rules.

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    stop_mult : float
        ATR multiplier for stop loss (default from config).
    tp_mult : float
        ATR multiplier for take profit (default from config).
    """

    def __init__(
        self,
        symbol: str = "UNKNOWN",
        stop_mult: float = ATR_STOP_MULT,
        tp_mult: float = ATR_TP_MULT,
        body_confirm: float = BODY_CONFIRM,
        vol_confirm: float = VOL_CONFIRM,
    ) -> None:
        self.symbol = symbol
        self.stop_mult = stop_mult
        self.tp_mult = tp_mult
        self.body_confirm = body_confirm
        self.vol_confirm = vol_confirm

    def generate(
        self,
        df: pd.DataFrame,
        regimes: pd.Series,
        lstm_probs: Optional[pd.Series] = None,
    ) -> List[Signal]:
        """
        Scan feature DataFrame and return a list of Signals.

        Parameters
        ----------
        df : DataFrame
            Output of DataLoader.compute_features().  Must contain:
            ema_fast, ema_slow, rsi14, atr14, close, session.
        regimes : Series
            HMM regime labels aligned with df.index (from RegimeHMM.predict).
        lstm_probs : Series, optional
            LSTM forward return probability (1 for up, 0 for down).
        """
        self._validate(df, regimes)
        signals: List[Signal] = []

        # Merge regime into the DataFrame for vectorised checks
        work = df.copy()
        work["regime"] = regimes

        # Previous-bar values for crossover and RSI extreme detection
        work["ema_fast_prev"] = work["ema_fast"].shift(1)
        work["ema_slow_prev"] = work["ema_slow"].shift(1)
        work["rsi14_prev"]    = work["rsi14"].shift(1)
        work["rsi14_prev2"]   = work["rsi14"].shift(2)  # 2 bars back for RSI confirm

        for i in range(2, len(work)):
            row = work.iloc[i]
            regime = row["regime"]

            # --- No trade in excluded regimes ---
            if regime in NO_TRADE_REGIMES:
                continue

            # --- Equity session filter: only 9:45–15:15 ET ---
            if self.symbol not in [GOLD_SYMBOL, GOLD_FUTURES_SYMBOL]:
                session = row.get("session", "regular")
                if session != "regular":
                    continue
                # Check intraday time within regular session
                ts = row.name
                if hasattr(ts, 'tz_convert'):
                    eastern = ts.tz_convert("US/Eastern")
                    hhmm = eastern.hour * 100 + eastern.minute
                    if hhmm < 945 or hhmm > 1515:
                        continue

            # --- ATR quality filter: skip dead/choppy bars ---
            atr = float(row["atr14"])
            price = float(row["close"])
            if atr < MIN_ATR_PCT * price:
                continue

            signal = None

            # --- Trending regime: EMA crossover (filtered by primary trend bias) ---
            if regime == TRENDING:
                signal = self._ema_crossover(row)

            # --- Ranging regime: RSI reversion ---
            # DISABLED: Empirically unprofitable (~20-25% WR) across all assets.
            # The 3:1 RR requires >25% WR to be profitable, and RSI reversion
            # barely reaches that level while generating many more false signals.
            # Only EMA crossovers in trending regime are retained.
            # elif regime == RANGING:
            #     signal = self._rsi_reversion(row)

            if signal is not None:
                # --- Phase 1: LSTM Confirmation Filter ---
                if lstm_probs is not None:
                    # Use label-based lookup (row.name is the bar's timestamp)
                    prob = lstm_probs.get(row.name, np.nan)
                    if not pd.isna(prob):
                        if signal.direction == "long" and prob < LSTM_PROB_THRESH:
                            signal = None
                        elif signal.direction == "short" and prob > LSTM_PROB_THRESH:
                            signal = None
                        else:
                            signal = Signal(**{**signal.to_dict(), "lstm_prob": round(float(prob), 4)})
                    # If prob is NaN (no LSTM coverage for this bar), let the signal through

            if signal is not None:
                signals.append(signal)

        log.info(
            "  Baseline signals for %s: %d generated (%d long, %d short)",
            self.symbol,
            len(signals),
            sum(1 for s in signals if s.direction == "long"),
            sum(1 for s in signals if s.direction == "short"),
        )
        return signals

    # ------------------------------------------------------------------
    # Signal rules
    # ------------------------------------------------------------------

    def _ema_crossover(self, row: pd.Series) -> Optional[Signal]:
        """EMA(9) crossing EMA(21) with volume, body conviction, and trend-bias filter.
        
        Trend-bias rule: only trade crossovers IN the direction of EMA200.
        - Bullish crossover (long) only when close > EMA200 (bull trend)
        - Bearish crossover (short) only when close < EMA200 (bear trend)
        This prevents shorting a rising market and longing a falling one.
        """
        fast_now  = row["ema_fast"]
        slow_now  = row["ema_slow"]
        fast_prev = row["ema_fast_prev"]
        slow_prev = row["ema_slow_prev"]

        if pd.isna(fast_prev) or pd.isna(slow_prev):
            return None

        # Volume confirmation: require above-average volume
        rel_vol = row.get("rel_volume", 1.0)
        if pd.isna(rel_vol) or rel_vol < self.vol_confirm:
            return None

        # Body conviction: strong candle, not a doji
        body_ratio = row.get("body_ratio", 1.0)
        if pd.isna(body_ratio) or body_ratio < self.body_confirm:
            return None

        # Trend bias: only trade with primary trend (EMA200 filter)
        trend_bias = row.get("trend_bias", np.nan)

        # Bullish crossover: fast crosses above slow — only in bull trend
        if fast_prev <= slow_prev and fast_now > slow_now:
            if pd.isna(trend_bias) or trend_bias == 1:  # Allow if no bias data
                return self._make_signal(row, "long", "ema_crossover", strength=0.7)

        # Bearish crossover: fast crosses below slow — only in bear trend
        if fast_prev >= slow_prev and fast_now < slow_now:
            if pd.isna(trend_bias) or trend_bias == 0:  # Allow if no bias data
                return self._make_signal(row, "short", "ema_crossover", strength=0.7)

        return None

    def _rsi_reversion(self, row: pd.Series) -> Optional[Signal]:
        """RSI mean-reversion in ranging regime with 2-bar extreme confirmation.
        
        Thresholds relaxed from 25/75 to 30/70 for more frequent quality signals.
        RSI must remain in extreme zone for >= 2 consecutive bars before reversion.
        """
        rsi       = row["rsi14"]
        rsi_prev  = row["rsi14_prev"]
        rsi_prev2 = row["rsi14_prev2"]

        if pd.isna(rsi_prev) or pd.isna(rsi_prev2):
            return None

        # Use 30/70 thresholds (more practical than 25/75 which rarely triggers)
        RSI_OB = 70.0
        RSI_OS = 30.0

        # Oversold → long: RSI was in extreme zone for ≥ 2 bars, now crosses back above
        if rsi_prev < RSI_OS and rsi_prev2 < RSI_OS and rsi >= RSI_OS:
            return self._make_signal(row, "long", "rsi_reversion", strength=0.65)

        # Overbought → short: RSI was in extreme zone for ≥ 2 bars, now crosses back below
        if rsi_prev > RSI_OB and rsi_prev2 > RSI_OB and rsi <= RSI_OB:
            return self._make_signal(row, "short", "rsi_reversion", strength=0.65)

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_signal(
        self,
        row: pd.Series,
        direction: str,
        rule: str,
        strength: float = 0.5,
    ) -> Signal:
        """Construct a Signal with ATR-based stops."""
        price = float(row["close"])
        atr   = float(row["atr14"])

        if direction == "long":
            stop = price - atr * self.stop_mult
            tp   = price + atr * self.tp_mult
        else:
            stop = price + atr * self.stop_mult
            tp   = price - atr * self.tp_mult

        return Signal(
            symbol=self.symbol,
            timestamp=str(row.name),
            direction=direction,
            strength=strength,
            regime=str(row["regime"]),
            session=str(row.get("session", "unknown")),
            entry_price=round(price, 4),
            stop_price=round(stop, 4),
            tp_price=round(tp, 4),
            atr=round(atr, 4),
            rule=rule,
        )

    def _validate(self, df: pd.DataFrame, regimes: pd.Series) -> None:
        required = ["ema_fast", "ema_slow", "rsi14", "atr14", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing signal columns: {missing}")
        if len(df) != len(regimes):
            raise ValueError(
                f"df length ({len(df)}) != regimes length ({len(regimes)})"
            )
