"""
live_trade_dia.py
=================
Live & Simulated Chronological Trader for Dow Jones (DIA).

Features:
  1. Feeds 1-minute bar data sequentially like a live real-time price feed.
  2. Strict 10.0% Max Drawdown Circuit Breaker (Kill Switch).
  3. Dynamic Risk Management:
     - Volatility & ATR-adjusted dynamic position sizing.
     - Dynamic drawdown de-risking tiers (throttles risk as DD deepens).
     - Dynamic Breakeven lock (+1.5x ATR) and Trailing Stop (+2.0x ATR).
  4. Real-time Mark-to-Market equity, High-Water Mark, and drawdown tracking.
  5. Multi-venue Smart Order Router execution with realistic slippage and venue fees.
  6. Modes:
     - Fast replay (default): processes all 1-min bars sequentially at full speed.
     - Streaming replay (--stream): animated live HUD dashboard with real-time bar feed.
     - Live Alpaca polling (--live): polls Alpaca API every minute for live bars.

Usage:
    # Full fast sequential replay with dynamic risk & 10% max DD limit
    python live_trade_dia.py

    # Streaming mode with live terminal HUD (e.g. 50 bars/sec)
    python live_trade_dia.py --stream --delay 0.02 --start 2026-06-01

    # Custom capital and risk
    python live_trade_dia.py --capital 25000 --risk-pct 1.0 --max-dd 10.0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR, CSV_PATHS
from ml_system.data.loader import DataLoader
from ml_system.data.alpaca_fetcher import AlpacaFetcher
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator, Signal
from ml_system.execution.dynamic_risk import DynamicRiskManager, Position

# Smart Order Router
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import SmartStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("live_trade_dia")

DIA_SYMBOL = "DIA"
ANCHOR_SYMBOLS = ["AAPL", "MSFT", "SPY"]


# ---------------------------------------------------------------------------
# Load Anchor HMMs
# ---------------------------------------------------------------------------

def load_anchor_models() -> dict[str, RegimeHMM]:
    models = {}
    for sym in ANCHOR_SYMBOLS:
        path = MODEL_DIR / f"hmm_{sym}_full.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {path}\n"
                f"Run 'python train_combined.py' first to build the models."
            )
        models[sym] = RegimeHMM.load(path)
    return models


def majority_vote_regimes(hmm_models: dict[str, RegimeHMM], df_features: pd.DataFrame) -> pd.Series:
    preds = {sym: hmm.predict(df_features) for sym, hmm in hmm_models.items()}
    pred_df = pd.DataFrame(preds, index=df_features.index)
    voted = pred_df.mode(axis=1)[0]
    voted.name = "regime"
    return voted


# ---------------------------------------------------------------------------
# Router Execution Helper
# ---------------------------------------------------------------------------

def route_order(
    symbol: str,
    side: OrderSide,
    qty: int,
    recent_bars: list[pd.Series],
) -> tuple[float, float, int]:
    """
    Submits a market order through the Smart Order Router across Alpha, Beta, Gamma.
    Returns (avg_fill_price, total_fees, total_filled).
    """
    if qty <= 0:
        return 0.0, 0.0, 0

    venues = {name: Exchange(name, cfg, symbol=symbol) for name, cfg in VENUE_CONFIGS.items()}
    order = Order(symbol=symbol, side=side, order_type=OrderType.MARKET, quantity=qty)

    bars_in = [
        Bar(symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for row in recent_bars
    ]

    result = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
    fill_price = result.avg_fill_price if result.total_filled > 0 else (recent_bars[-1].close if recent_bars else 0.0)
    return fill_price, result.total_fees, result.total_filled


def levels_from_actual_fill(signal: Signal, fill_price: float) -> tuple[float, float]:
    """Keep the signal's ATR risk/reward distances after a delayed fill."""
    stop_distance = abs(signal.entry_price - signal.stop_price)
    target_distance = abs(signal.tp_price - signal.entry_price)
    if signal.direction == "long":
        return fill_price - stop_distance, fill_price + target_distance
    return fill_price + stop_distance, fill_price - target_distance


# ---------------------------------------------------------------------------
# Chronological Live Price Simulation Engine
# ---------------------------------------------------------------------------

def run_chronological_simulation(
    capital: float = 10_000.0,
    max_dd: float = 10.0,
    risk_pct: float = 1.0,
    stream: bool = False,
    delay: float = 0.02,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:

    log.info("=" * 75)
    log.info("CHRONOLOGICAL LIVE-PRICE SIMULATION — DOW JONES (DIA)")
    log.info("Capital: $%.2f | Max DD Limit: %.1f%% | Base Risk: %.1f%%", capital, max_dd, risk_pct)
    log.info("Anchor Brain: AAPL + MSFT + SPY HMMs (majority vote)")
    log.info("=" * 75)

    # 1. Load Data
    csv_path = CSV_PATHS.get(DIA_SYMBOL)
    if not csv_path or not Path(csv_path).exists():
        log.info("Fetching DIA 1-min data from Alpaca API...")
        fetcher = AlpacaFetcher()
        fetcher.fetch(DIA_SYMBOL, datetime(2024, 1, 1, tzinfo=timezone.utc), datetime.now(timezone.utc), 1)

    loader = DataLoader(DIA_SYMBOL)
    df_1min = loader.load()

    # Optional date filters
    if start_date:
        df_1min = df_1min[df_1min.index >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df_1min = df_1min[df_1min.index <= pd.to_datetime(end_date, utc=True)]

    log.info("Processing %d 1-min bars (%s -> %s)", len(df_1min), df_1min.index[0].date(), df_1min.index[-1].date())

    # 2. Resample and compute features
    loader.set_dataframe(df_1min)
    df_15min = loader.resample(DEFAULT_TIMEFRAME)
    df_features = loader.compute_features(df_15min)

    # 3. Model regimes & baseline signal generator
    models = load_anchor_models()
    regimes = majority_vote_regimes(models, df_features)
    generator = BaselineSignalGenerator(DIA_SYMBOL)
    all_signals = generator.generate(df_features, regimes)

    # Map signals by timestamp of the 15-min bar completion
    signal_map: dict[pd.Timestamp, Signal] = {
        pd.to_datetime(s.timestamp, utc=True): s for s in all_signals
    }
    log.info("Generated %d potential signals across the dataset", len(signal_map))

    # 4. Initialize Dynamic Risk Manager
    risk_mgr = DynamicRiskManager(
        initial_capital=capital,
        max_drawdown_pct=max_dd,
        base_risk_pct=risk_pct,
    )

    open_pos: Optional[Position] = None
    pending_signal: Optional[Signal] = None  # staged signal — entered on the NEXT bar
    recent_bars_buffer: list[pd.Series] = []
    trade_history: list[dict] = []
    dynamic_events_count = {"breakeven": 0, "trailing": 0, "kill_switch": 0}

    log.info("-" * 80)
    log.info("%-19s %-6s %-12s %-5s %-9s %-9s %-6s %-10s %-10s",
             "Exit Timestamp", "Dir", "Rule", "Qty", "Entry", "Exit", "Type", "Net PnL", "Equity")
    log.info("-" * 80)

    # 5. Chronological bar-by-bar live streaming loop
    total_bars = len(df_1min)
    t_start = time.time()

    for idx, (bar_ts, row) in enumerate(df_1min.iterrows()):
        recent_bars_buffer.append(row)
        if len(recent_bars_buffer) > 10:
            recent_bars_buffer.pop(0)

        current_price = float(row.close)
        high = float(row.high)
        low = float(row.low)

        # -------------------------------------------------------------------
        # A. Mark-to-market & 10% Max Drawdown check
        # -------------------------------------------------------------------
        equity = risk_mgr.mark_to_market(current_price, open_pos)

        # Check circuit breaker
        if risk_mgr.is_kill_switch_tripped:
            dynamic_events_count["kill_switch"] += 1
            if open_pos is not None:
                # Emergency close open position on kill switch trip
                fill_price, exit_fees, _ = route_order(
                    symbol=DIA_SYMBOL,
                    side=OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                    qty=open_pos.qty,
                    recent_bars=recent_bars_buffer,
                )
                gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                        else (open_pos.entry_price - fill_price) * open_pos.qty
                net_pnl = gross - open_pos.entry_cost - exit_fees
                risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)

                trade_history.append({
                    "entry_time": open_pos.entry_time,
                    "exit_time": str(bar_ts),
                    "direction": open_pos.direction,
                    "rule": open_pos.rule,
                    "qty": open_pos.qty,
                    "entry_price": open_pos.entry_price,
                    "exit_price": fill_price,
                    "outcome": "KILL_SWITCH",
                    "net_pnl": round(net_pnl, 2),
                    "equity": round(risk_mgr.equity, 2),
                    "drawdown_pct": round(risk_mgr.current_drawdown_pct, 2),
                })
                open_pos = None
                log.warning("[%s] KILL SWITCH: Emergency closed position! Trading halted.", str(bar_ts)[:19])

        # -------------------------------------------------------------------
        # B. Check open position (SL / TP / Trailing Stop)
        # -------------------------------------------------------------------
        if open_pos is not None and not risk_mgr.is_kill_switch_tripped:
            # 1. Update dynamic trailing stop & breakeven
            new_stop, update_event = risk_mgr.update_stops_and_targets(open_pos, high, low, current_price)
            if update_event == "breakeven_locked":
                dynamic_events_count["breakeven"] += 1
                if stream:
                    log.info("  [%s] Dynamic Risk: Stop moved to BREAKEVEN ($%.2f)", str(bar_ts)[:19], new_stop)
            elif update_event in ("trailing_stop_raised", "trailing_stop_lowered"):
                dynamic_events_count["trailing"] += 1

            # 2. Check for Exit on current 1-min bar
            exit_triggered = False
            outcome = ""
            raw_exit_price = current_price

            if open_pos.direction == "long":
                if low <= open_pos.current_stop:
                    raw_exit_price = open_pos.current_stop
                    outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                    exit_triggered = True
                elif high >= open_pos.tp_price:
                    raw_exit_price = open_pos.tp_price
                    outcome = "TP"
                    exit_triggered = True
            else:  # short
                if high >= open_pos.current_stop:
                    raw_exit_price = open_pos.current_stop
                    outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                    exit_triggered = True
                elif low <= open_pos.tp_price:
                    raw_exit_price = open_pos.tp_price
                    outcome = "TP"
                    exit_triggered = True

            if exit_triggered:
                # Execute exit through OrderRouter
                fill_price, exit_fees, _ = route_order(
                    symbol=DIA_SYMBOL,
                    side=OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                    qty=open_pos.qty,
                    recent_bars=recent_bars_buffer,
                )
                if fill_price == 0:
                    fill_price = raw_exit_price

                gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                        else (open_pos.entry_price - fill_price) * open_pos.qty
                net_pnl = gross - open_pos.entry_cost - exit_fees

                risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)

                trade_history.append({
                    "entry_time": open_pos.entry_time,
                    "exit_time": str(bar_ts),
                    "direction": open_pos.direction,
                    "rule": open_pos.rule,
                    "qty": open_pos.qty,
                    "entry_price": open_pos.entry_price,
                    "exit_price": fill_price,
                    "outcome": outcome,
                    "net_pnl": round(net_pnl, 2),
                    "equity": round(risk_mgr.equity, 2),
                    "drawdown_pct": round(risk_mgr.current_drawdown_pct, 2),
                })

                pnl_str = f"+${net_pnl:,.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):,.2f}"
                log.info(
                    "%-19s %-6s %-12s %-5d $%-8.2f $%-8.2f %-6s %-10s $%-10.2f",
                    str(bar_ts)[:19],
                    open_pos.direction.upper(),
                    open_pos.rule[:12],
                    open_pos.qty,
                    open_pos.entry_price,
                    fill_price,
                    outcome,
                    pnl_str,
                    risk_mgr.equity,
                )
                open_pos = None

        # -------------------------------------------------------------------
        # C. Stage signal at 15-min bar close; enter on the NEXT 1-min bar
        # -------------------------------------------------------------------
        if bar_ts in signal_map and open_pos is None and pending_signal is None:
            pending_signal = signal_map[bar_ts]

        # Execute pending signal on the first bar strictly AFTER signal close
        if open_pos is None and pending_signal is not None and not risk_mgr.is_kill_switch_tripped:
            sig = pending_signal
            sig_ts = pd.to_datetime(sig.timestamp, utc=True)
            if bar_ts > sig_ts:
                # Dynamic Position Sizing
                shares = risk_mgr.calculate_position_size(
                    entry_price=sig.entry_price,
                    stop_price=sig.stop_price,
                    regime=sig.regime,
                    atr=sig.atr,
                )

                if shares > 0:
                    # Submit entry order to Smart Order Router
                    fill_price, entry_fees, filled_qty = route_order(
                        symbol=DIA_SYMBOL,
                        side=OrderSide.BUY if sig.direction == "long" else OrderSide.SELL,
                        qty=shares,
                        recent_bars=recent_bars_buffer,
                    )
                    if filled_qty > 0:
                        open_pos = Position(
                            symbol=DIA_SYMBOL,
                            direction=sig.direction,
                            entry_time=str(bar_ts),
                            entry_price=fill_price,
                            qty=filled_qty,
                            initial_stop=sig.stop_price,
                            current_stop=sig.stop_price,
                            tp_price=sig.tp_price,
                            atr=sig.atr,
                            rule=sig.rule,
                            regime=sig.regime,
                            entry_cost=entry_fees,
                        )
                        if stream:
                            log.info("  [%s] NEW POSITION: %s %d DIA @ $%.2f | Stop: $%.2f | TP: $%.2f | DD: %.1f%%",
                                     str(bar_ts)[:19], sig.direction.upper(), filled_qty, fill_price,
                                     sig.stop_price, sig.tp_price, risk_mgr.current_drawdown_pct)
                pending_signal = None  # consumed

        # -------------------------------------------------------------------
        # D. Streaming HUD Display
        # -------------------------------------------------------------------
        if stream:
            time.sleep(delay)
            unrealized = 0.0
            pos_desc = "FLAT"
            if open_pos is not None:
                if open_pos.direction == "long":
                    unrealized = (current_price - open_pos.entry_price) * open_pos.qty
                else:
                    unrealized = (open_pos.entry_price - current_price) * open_pos.qty
                pos_desc = f"{open_pos.direction.upper()} {open_pos.qty}x@${open_pos.entry_price:.2f} (uPnL: ${unrealized:+.2f})"

            dd_bar = "#" * int(risk_mgr.current_drawdown_pct) + "-" * (10 - int(risk_mgr.current_drawdown_pct))
            sys.stdout.write(
                f"\r[{str(bar_ts)[:19]}] DIA: ${current_price:6.2f} | Eq: ${risk_mgr.equity:8.2f} | "
                f"HWM: ${risk_mgr.high_water_mark:8.2f} | DD: [{dd_bar}] {risk_mgr.current_drawdown_pct:4.1f}%/10% | {pos_desc:35s}"
            )
            sys.stdout.flush()

    if stream:
        print()

    # 6. Final Performance Report
    n = len(trade_history)
    wins = sum(1 for t in trade_history if t["net_pnl"] > 0)
    losses = sum(1 for t in trade_history if t["net_pnl"] < 0)
    win_rate = (wins / n * 100) if n else 0.0
    net_pnl = risk_mgr.equity - capital
    roi_pct = (net_pnl / capital) * 100
    gross_gains = sum(t["net_pnl"] for t in trade_history if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trade_history if t["net_pnl"] < 0))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else float("inf")

    log.info("=" * 75)
    log.info("PERFORMANCE REPORT — DOW JONES (DIA) WITH DYNAMIC RISK")
    log.info("=" * 75)
    log.info("  Initial Capital       : $%.2f", capital)
    log.info("  Ending Equity         : $%.2f", risk_mgr.equity)
    log.info("  Net Realized PnL      : +$%.2f (+%.2f%% ROI)" if net_pnl >= 0 else "  Net Realized PnL      : -$%.2f (%.2f%% ROI)", abs(net_pnl), roi_pct)
    log.info("  High-Water Mark (HWM) : $%.2f", risk_mgr.high_water_mark)
    log.info("  Peak Drawdown         : %.2f%% (Hard Limit: %.1f%% - %s)",
             risk_mgr.peak_drawdown_pct, max_dd,
             "PASSED [DD < 10%]" if risk_mgr.peak_drawdown_pct < max_dd else "TRIPPED KILL SWITCH")
    log.info("  Total Closed Trades   : %d", n)
    log.info("  Wins / Losses         : %d / %d (Win Rate: %.1f%%)", wins, losses, win_rate)
    log.info("  Gross Gains / Losses  : +$%.2f / -$%.2f", gross_gains, gross_losses)
    log.info("  Profit Factor         : %.2f", profit_factor)
    log.info("  Total Router Fees     : $%.2f", risk_mgr.total_fees_paid)
    log.info("  --- Dynamic Risk Events ---")
    log.info("  Breakeven Locks       : %d trades protected at breakeven", dynamic_events_count["breakeven"])
    log.info("  Trailing Stops Raised : %d times", dynamic_events_count["trailing"])
    log.info("  10%% Kill Switch Status : %s", "ACTIVE / TRIPPED" if risk_mgr.is_kill_switch_tripped else "SAFE / ACTIVE")
    log.info("=" * 75)

    # 7. Save trade log
    out_path = Path("results") / "dia_live_trades.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(trade_history).to_csv(out_path, index=False)
    log.info("Trade history and equity curve saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Real-Time Live Alpaca Polling Mode
# ---------------------------------------------------------------------------

def run_live_alpaca_polling(
    capital: float = 10_000.0,
    max_dd: float = 10.0,
    risk_pct: float = 1.0,
) -> None:
    """
    Connects to Alpaca API in real-time, fetching incoming 1-min bars every minute.
    Maintains a rolling 1-min buffer, resamples to 15-min bars on each tick,
    generates signals using the anchor HMM brain, and executes trades through the
    Smart Order Router with full 10% drawdown circuit breaker and trailing stop logic.
    """
    log.info("=" * 75)
    log.info("LIVE REAL-TIME ALPACA TRADER — DOW JONES (DIA)")
    log.info("Capital: $%.2f | Max DD: %.1f%% | Dynamic Risk: %.1f%%", capital, max_dd, risk_pct)
    log.info("=" * 75)

    fetcher = AlpacaFetcher()
    risk_mgr = DynamicRiskManager(
        initial_capital=capital,
        max_drawdown_pct=max_dd,
        base_risk_pct=risk_pct,
    )
    models = load_anchor_models()
    generator = BaselineSignalGenerator(DIA_SYMBOL)

    # --- Warm up: prime the 1-min buffer with recent history ---
    log.info("Warming up 1-minute historical buffer (5 days)...")
    now = datetime.now(timezone.utc)
    start_warmup = now - pd.Timedelta(days=5)
    df_buffer = fetcher.fetch(DIA_SYMBOL, start_warmup, now, timeframe_minutes=1)
    log.info("Buffer primed with %d 1-min bars.", len(df_buffer))

    last_processed_ts = df_buffer.index[-1] if not df_buffer.empty else None
    open_pos: Optional[Position] = None
    pending_signal: Optional[Signal] = None
    recent_bars_buffer: list[pd.Series] = []
    # Pre-fill recent bars buffer from warmup
    for _, row in df_buffer.tail(10).iterrows():
        recent_bars_buffer.append(row)

    # Track the last 15-min signal timestamp we acted on (avoid duplicate signals)
    last_signal_ts: Optional[pd.Timestamp] = None

    log.info("Starting live polling loop (press Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(15)  # Poll every 15s for the closed 1-min bar
            current_time = datetime.now(timezone.utc)
            fetch_start = current_time - pd.Timedelta(minutes=5)
            latest_bars = fetcher._fetch_alpaca(DIA_SYMBOL, fetch_start, current_time, 1)

            if latest_bars.empty:
                continue

            new_bars = latest_bars[latest_bars.index > last_processed_ts] if last_processed_ts else latest_bars
            if new_bars.empty:
                continue

            for bar_ts, row in new_bars.iterrows():
                last_processed_ts = bar_ts

                # Append to rolling buffer
                df_buffer = pd.concat([df_buffer, pd.DataFrame([row], index=[bar_ts])])
                recent_bars_buffer.append(row)
                if len(recent_bars_buffer) > 10:
                    recent_bars_buffer.pop(0)

                current_price = float(row.close)
                high = float(row.high)
                low = float(row.low)

                # Mark to market & Drawdown check
                equity = risk_mgr.mark_to_market(current_price, open_pos)

                # Kill switch
                if risk_mgr.is_kill_switch_tripped:
                    log.warning("[%s] KILL SWITCH TRIPPED! DD %.2f%% >= %.1f%%. Halting all new orders.",
                                str(bar_ts)[:19], risk_mgr.current_drawdown_pct, max_dd)
                    if open_pos is not None:
                        fill_price, exit_fees, _ = route_order(
                            symbol=DIA_SYMBOL,
                            side=OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                            qty=open_pos.qty,
                            recent_bars=recent_bars_buffer,
                        )
                        gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                                else (open_pos.entry_price - fill_price) * open_pos.qty
                        net_pnl = gross - open_pos.entry_cost - exit_fees
                        risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)
                        log.warning("[%s] Emergency exit at $%.2f. Net PnL: $%.2f",
                                    str(bar_ts)[:19], fill_price, net_pnl)
                        open_pos = None
                    pending_signal = None
                    continue

                # Manage open position: trailing stop & SL/TP
                if open_pos is not None:
                    new_stop, update_event = risk_mgr.update_stops_and_targets(open_pos, high, low, current_price)
                    if update_event == "breakeven_locked":
                        log.info("[%s] Dynamic Risk: Stop moved to Breakeven @ $%.2f", str(bar_ts)[:19], new_stop)
                    elif update_event in ("trailing_stop_raised", "trailing_stop_lowered"):
                        log.info("[%s] Dynamic Risk: Trailing Stop adjusted to $%.2f", str(bar_ts)[:19], new_stop)

                    exit_triggered = False
                    outcome = ""
                    raw_exit_price = current_price
                    if open_pos.direction == "long":
                        if low <= open_pos.current_stop:
                            raw_exit_price = open_pos.current_stop
                            outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                            exit_triggered = True
                        elif high >= open_pos.tp_price:
                            raw_exit_price = open_pos.tp_price
                            outcome = "TP"
                            exit_triggered = True
                    else:
                        if high >= open_pos.current_stop:
                            raw_exit_price = open_pos.current_stop
                            outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                            exit_triggered = True
                        elif low <= open_pos.tp_price:
                            raw_exit_price = open_pos.tp_price
                            outcome = "TP"
                            exit_triggered = True

                    if exit_triggered:
                        fill_price, exit_fees, _ = route_order(
                            symbol=DIA_SYMBOL,
                            side=OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                            qty=open_pos.qty,
                            recent_bars=recent_bars_buffer,
                        )
                        if fill_price == 0:
                            fill_price = raw_exit_price
                        gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                                else (open_pos.entry_price - fill_price) * open_pos.qty
                        net_pnl = gross - open_pos.entry_cost - exit_fees
                        risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)
                        log.info("[%s] EXIT %s %s %dx @ $%.2f | %s PnL: $%.2f | Equity: $%.2f",
                                 str(bar_ts)[:19], outcome, open_pos.direction.upper(),
                                 open_pos.qty, fill_price, outcome,
                                 net_pnl, risk_mgr.equity)
                        open_pos = None

                # Generate fresh signals at 15-min bar boundaries only.
                # The signal generator needs >= 3 bars for shift() to work; passing
                # only iloc[-1:] always produces zero signals (loop starts at i=2).
                # Detect boundary: this 1-min bar minute is a multiple of 15.
                at_15min_boundary = (bar_ts.minute % 15 == 0)
                if at_15min_boundary and open_pos is None and not risk_mgr.is_kill_switch_tripped:
                    loader_live = DataLoader(DIA_SYMBOL)
                    loader_live.set_dataframe(df_buffer)
                    try:
                        df_15min_live = loader_live.resample("15min")
                        df_feat_live  = loader_live.compute_features(df_15min_live)
                        if len(df_feat_live) >= 20:  # need warmup bars for indicators
                            regimes_live = majority_vote_regimes(models, df_feat_live)
                            # Run generator over all features so the shift() windows work;
                            # deduplicate via last_signal_ts so stale signals are ignored.
                            new_signals = generator.generate(df_feat_live, regimes_live)
                            for sig in new_signals:
                                sig_ts = pd.to_datetime(sig.timestamp, utc=True)
                                if last_signal_ts is None or sig_ts > last_signal_ts:
                                    pending_signal = sig
                                    last_signal_ts = sig_ts
                                    log.info("[%s] SIGNAL: %s %s (Rule: %s | Entry: $%.2f | Stop: $%.2f | TP: $%.2f)",
                                             str(bar_ts)[:19], sig.direction.upper(), DIA_SYMBOL,
                                             sig.rule, sig.entry_price, sig.stop_price, sig.tp_price)
                    except Exception as e:
                        log.debug("Live feature computation skipped: %s", e)

                # Execute pending signal -- only on bars at or after the 15-min bar close.
                # sig.timestamp is the bar open label; bar closes at sig_ts + 15 min.
                # This mirrors the lookahead guard used in run_chronological_simulation.
                if open_pos is None and pending_signal is not None and not risk_mgr.is_kill_switch_tripped:
                    sig = pending_signal
                    sig_ts    = pd.to_datetime(sig.timestamp, utc=True)
                    bar_close = sig_ts + pd.Timedelta(minutes=15)
                    if bar_ts >= bar_close:
                        shares = risk_mgr.calculate_position_size(
                            entry_price=sig.entry_price,
                            stop_price=sig.stop_price,
                            regime=sig.regime,
                            atr=sig.atr,
                        )
                        if shares > 0:
                            fill_price, entry_fees, filled_qty = route_order(
                                symbol=DIA_SYMBOL,
                                side=OrderSide.BUY if sig.direction == "long" else OrderSide.SELL,
                                qty=shares,
                                recent_bars=recent_bars_buffer,
                            )
                            if filled_qty > 0:
                                open_pos = Position(
                                    symbol=DIA_SYMBOL, direction=sig.direction,
                                    entry_time=str(bar_ts), entry_price=fill_price,
                                    qty=filled_qty, initial_stop=sig.stop_price,
                                    current_stop=sig.stop_price, tp_price=sig.tp_price,
                                    atr=sig.atr, rule=sig.rule, regime=sig.regime,
                                    entry_cost=entry_fees,
                                )
                                log.info("[%s] ENTRY: %s %d DIA @ $%.2f | Stop: $%.2f | TP: $%.2f | Equity: $%.2f",
                                         str(bar_ts)[:19], sig.direction.upper(), filled_qty,
                                         fill_price, sig.stop_price, sig.tp_price, risk_mgr.equity)
                        pending_signal = None  # consumed regardless (avoid stale signal)

                # Status log
                pos_str = f"{open_pos.direction.upper()} {open_pos.qty}x@${open_pos.entry_price:.2f}" if open_pos else "FLAT"
                log.info(
                    "[%s] DIA: $%.2f | Equity: $%.2f | HWM: $%.2f | DD: %.2f%%/%.1f%% | %s",
                    str(bar_ts)[:19], current_price, equity, risk_mgr.high_water_mark,
                    risk_mgr.current_drawdown_pct, max_dd, pos_str
                )

    except KeyboardInterrupt:
        log.info("Live polling stopped by user. Final equity: $%.2f", risk_mgr.equity)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live & Chronological Trader for Dow Jones (DIA) with Dynamic Risk Management.")
    parser.add_argument("--capital",   type=float, default=10_000.0, help="Initial account equity (default $10,000)")
    parser.add_argument("--max-dd",    type=float, default=10.0,     help="Maximum allowable drawdown % (default 10.0%)")
    parser.add_argument("--risk-pct",  type=float, default=1.0,      help="Base risk budget % per trade (default 1.0%)")
    parser.add_argument("--stream",    action="store_true",          help="Run animated live terminal streaming HUD")
    parser.add_argument("--delay",     type=float, default=0.01,     help="Seconds delay between bars in stream mode")
    parser.add_argument("--live",      action="store_true",          help="Poll Alpaca API in real-time live trading mode")
    parser.add_argument("--start",     type=str,   default=None,     help="Filter start date e.g. 2025-01-01")
    parser.add_argument("--end",       type=str,   default=None,     help="Filter end date e.g. 2026-09-27")
    args = parser.parse_args()

    if args.live:
        run_live_alpaca_polling(
            capital=args.capital,
            max_dd=args.max_dd,
            risk_pct=args.risk_pct,
        )
    else:
        run_chronological_simulation(
            capital=args.capital,
            max_dd=args.max_dd,
            risk_pct=args.risk_pct,
            stream=args.stream,
            delay=args.delay,
            start_date=args.start,
            end_date=args.end,
        )
