"""
Dynamic Risk Manager — Adaptive Position Sizing & Drawdown Protection
====================================================================
Implements institutional-grade risk management:
1. Strict 10.0% Max Drawdown Circuit Breaker (Kill Switch).
2. Volatility-adjusted position sizing based on account equity and ATR stop distance.
3. Drawdown-adaptive de-risking tiers (reduces position size as drawdown deepens).
4. Regime-adaptive sizing (adjusts risk based on trending vs ranging regimes).
5. Dynamic Trailing Stop & Breakeven:
   - Moves stop to Breakeven (+ spread buffer) once profit reaches +1.5x ATR.
   - Trails stop at 1.0x ATR behind highest price once profit reaches +2.0x ATR.
6. Real-time portfolio equity, high-water mark (HWM), and drawdown tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Literal

log = logging.getLogger("dynamic_risk")


@dataclass
class Position:
    symbol: str
    direction: Literal["long", "short"]
    entry_time: str
    entry_price: float
    qty: int
    initial_stop: float
    current_stop: float
    tp_price: float
    atr: float
    rule: str
    regime: str
    entry_cost: float = 0.0
    highest_favorable_price: float = 0.0
    lowest_favorable_price: float = 0.0
    is_breakeven_set: bool = False
    is_trailing_active: bool = False

    def __post_init__(self):
        if self.highest_favorable_price == 0.0:
            self.highest_favorable_price = self.entry_price
        if self.lowest_favorable_price == 0.0:
            self.lowest_favorable_price = self.entry_price


class DynamicRiskManager:
    """
    Manages account equity, dynamic risk sizing, trailing stops,
    and enforces a strict 10% maximum drawdown circuit breaker.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        max_drawdown_pct: float = 10.0,
        base_risk_pct: float = 1.0,
        max_notional_pct: float = 50.0,
    ) -> None:
        self.initial_capital: float = float(initial_capital)
        self.current_cash: float = float(initial_capital)
        self.equity: float = float(initial_capital)
        self.high_water_mark: float = float(initial_capital)
        self.max_drawdown_pct: float = float(max_drawdown_pct)
        self.base_risk_pct: float = float(base_risk_pct)
        self.max_notional_pct: float = float(max_notional_pct)

        # Drawdown tracking
        self.current_drawdown_pct: float = 0.0
        self.peak_drawdown_pct: float = 0.0
        self.is_kill_switch_tripped: bool = False

        # Metrics
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_fees_paid: float = 0.0

    # -----------------------------------------------------------------------
    # Mark to Market & Drawdown Checking
    # -----------------------------------------------------------------------

    def mark_to_market(
        self,
        current_price: float,
        position: Optional[Position] = None,
    ) -> float:
        """
        Update account equity with unrealized PnL from an open position.
        Checks and updates drawdown against the 10% circuit breaker.
        """
        unrealized = 0.0
        if position is not None and position.qty > 0:
            if position.direction == "long":
                unrealized = (current_price - position.entry_price) * position.qty
            else:
                unrealized = (position.entry_price - current_price) * position.qty

        self.equity = self.current_cash + unrealized

        # Update High-Water Mark
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        # Calculate current drawdown from HWM
        if self.high_water_mark > 0:
            self.current_drawdown_pct = max(
                0.0, (self.high_water_mark - self.equity) / self.high_water_mark * 100.0
            )

        if self.current_drawdown_pct > self.peak_drawdown_pct:
            self.peak_drawdown_pct = self.current_drawdown_pct

        # Check circuit breaker
        if self.current_drawdown_pct >= self.max_drawdown_pct and not self.is_kill_switch_tripped:
            self.is_kill_switch_tripped = True
            log.warning(
                "*** CIRCUIT BREAKER TRIPPED! *** Drawdown %.2f%% reached max limit %.2f%%. "
                "Halting all new risk!",
                self.current_drawdown_pct,
                self.max_drawdown_pct,
            )

        return self.equity

    # -----------------------------------------------------------------------
    # Dynamic Sizing & De-risking Tiers
    # -----------------------------------------------------------------------

    def get_drawdown_throttle(self) -> float:
        """
        Dynamically throttles risk budget as account drawdown approaches 10%:
          - DD < 3.0%:  1.00x normal risk
          - DD 3% - 6%: 0.75x normal risk
          - DD 6% - 8%: 0.50x normal risk
          - DD 8% - 10%: 0.25x normal risk (capital preservation)
          - DD >= 10%:  0.00x (Kill Switch)
        """
        if self.is_kill_switch_tripped or self.current_drawdown_pct >= self.max_drawdown_pct:
            return 0.0
        elif self.current_drawdown_pct < 3.0:
            return 1.0
        elif self.current_drawdown_pct < 6.0:
            return 0.75
        elif self.current_drawdown_pct < 8.0:
            return 0.50
        else:
            return 0.25

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        regime: str,
        atr: float,
    ) -> int:
        """
        Dynamically calculate share quantity:
        1. Multiplier for drawdown throttle (de-risking).
        2. Multiplier for market regime confidence.
        3. Dollar risk = Equity * base_risk_pct * multipliers.
        4. Shares = Dollar risk / stop_distance.
        5. Hard caps for single-position notional size and account cash.
        """
        if self.is_kill_switch_tripped:
            log.info("  Kill switch is ACTIVE (DD >= %.1f%%) — 0 shares allocated.", self.max_drawdown_pct)
            return 0

        dd_multiplier = self.get_drawdown_throttle()
        if dd_multiplier <= 0.0:
            return 0

        # Regime confidence multiplier
        regime_multiplier = 1.0
        if regime == "trending":
            regime_multiplier = 1.10  # higher confidence in trending regime
        elif regime == "ranging":
            regime_multiplier = 0.85  # tighter risk in mean-reversion
        elif regime in ("high_vol_breakout", "low_liquidity"):
            regime_multiplier = 0.50

        # Dollar risk budget
        risk_budget = (
            self.equity * (self.base_risk_pct / 100.0) * dd_multiplier * regime_multiplier
        )

        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0

        # Raw volatility-based share size
        qty = int(risk_budget / stop_distance)
        if qty < 1:
            qty = 1  # Minimum 1 share if risk budget allows

        # Max notional cap (e.g. 50% of account equity)
        max_notional_cash = self.equity * (self.max_notional_pct / 100.0)
        max_shares = int(max_notional_cash / entry_price) if entry_price > 0 else 0
        qty = min(qty, max_shares)

        # Cash check
        if qty * entry_price > self.current_cash:
            qty = int(self.current_cash / entry_price)

        return max(0, qty)

    # -----------------------------------------------------------------------
    # Dynamic Trailing Stop & Breakeven
    # -----------------------------------------------------------------------

    def update_stops_and_targets(
        self,
        pos: Position,
        high: float,
        low: float,
        close: float,
    ) -> tuple[float, Optional[str]]:
        """
        Dynamically adjusts stop-loss based on price progression:
        1. Breakeven: when profit >= +1.5x ATR, move stop to entry + buffer (free trade).
        2. Trailing Stop: when profit >= +2.0x ATR, trail 1.0x ATR behind peak price.
        Returns: (new_stop, trigger_event)
        """
        atr = pos.atr
        event = None

        if pos.direction == "long":
            if high > pos.highest_favorable_price:
                pos.highest_favorable_price = high

            favorable_move = pos.highest_favorable_price - pos.entry_price

            # Trailing Stop check (Profit >= 2.0x ATR)
            if favorable_move >= 2.0 * atr:
                trail_stop = round(pos.highest_favorable_price - 1.0 * atr, 4)
                if trail_stop > pos.current_stop:
                    pos.current_stop = trail_stop
                    pos.is_trailing_active = True
                    event = "trailing_stop_raised"

            # Breakeven check (Profit >= 1.5x ATR)
            elif favorable_move >= 1.5 * atr and not pos.is_breakeven_set:
                be_stop = round(pos.entry_price + 0.05, 4)  # entry + 5c spread buffer
                if be_stop > pos.current_stop:
                    pos.current_stop = be_stop
                    pos.is_breakeven_set = True
                    event = "breakeven_locked"

        else:  # short
            if low < pos.lowest_favorable_price:
                pos.lowest_favorable_price = low

            favorable_move = pos.entry_price - pos.lowest_favorable_price

            # Trailing Stop check (Profit >= 2.0x ATR)
            if favorable_move >= 2.0 * atr:
                trail_stop = round(pos.lowest_favorable_price + 1.0 * atr, 4)
                if trail_stop < pos.current_stop:
                    pos.current_stop = trail_stop
                    pos.is_trailing_active = True
                    event = "trailing_stop_lowered"

            # Breakeven check (Profit >= 1.5x ATR)
            elif favorable_move >= 1.5 * atr and not pos.is_breakeven_set:
                be_stop = round(pos.entry_price - 0.05, 4)
                if be_stop < pos.current_stop:
                    pos.current_stop = be_stop
                    pos.is_breakeven_set = True
                    event = "breakeven_locked"

        return pos.current_stop, event

    # -----------------------------------------------------------------------
    # Trade Accounting
    # -----------------------------------------------------------------------

    def close_position(
        self,
        net_pnl: float,
        fees: float,
    ) -> None:
        """Record trade realization and update cash and equity."""
        self.current_cash += net_pnl
        self.equity = self.current_cash
        self.total_fees_paid += fees
        self.total_trades += 1

        if net_pnl > 0:
            self.winning_trades += 1
        elif net_pnl < 0:
            self.losing_trades += 1

        # Update HWM and Drawdown on realized equity
        if self.equity > self.high_water_mark:
            self.high_water_mark = self.equity

        if self.high_water_mark > 0:
            self.current_drawdown_pct = max(
                0.0, (self.high_water_mark - self.equity) / self.high_water_mark * 100.0
            )

        if self.current_drawdown_pct > self.peak_drawdown_pct:
            self.peak_drawdown_pct = self.current_drawdown_pct
