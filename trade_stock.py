"""
trade_stock.py
==============
Universal Stock Trader using the Anchor HMM Brain (AAPL + MSFT + SPY).

Works for ANY US equity symbol (e.g. NVDA, TSLA, AMD, QQQ, DIA, MSFT, AAPL, SPY).
- Automatically fetches 1-min data from Alpaca if missing.
- Supports both risk modes:
    1. 'previous': Fixed share sizing (default 500), pure 3:1 R:R.
    2. 'dynamic' : Volatility-adjusted sizing (1% equity risk), 10% max DD kill switch,
                   dynamic breakeven lock (+1.5 ATR) and trailing stop (+2.0 ATR).
- Simulates realistic multi-venue fills, slippage, and exchange fees via Smart Order Router.
- Outputs trade log to results/{symbol}_trades.csv.

Usage:
    # Trade NVDA with previous risk management (500 shares, pure 3:1 R:R)
    python trade_stock.py --symbol NVDA

    # Trade TSLA with dynamic risk management (10% max DD, trailing stop)
    python trade_stock.py --symbol TSLA --risk dynamic

    # Force fresh data fetch from Alpaca
    python trade_stock.py --symbol AMD --fetch
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR, CSV_PATHS, DATA_DIR
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
log = logging.getLogger("trade_stock")

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
# Router Execution
# ---------------------------------------------------------------------------

def route_order(
    symbol: str,
    side: OrderSide,
    qty: int,
    recent_bars: list[pd.Series],
) -> tuple[float, float, int]:
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


# ---------------------------------------------------------------------------
# Mode 1: Previous Risk Management (Static Sizing, Pure 3:1 R:R)
# ---------------------------------------------------------------------------

def run_previous_risk(symbol: str, df_1min: pd.DataFrame, signals: list[Signal], qty: int = 500) -> None:
    log.info("-" * 75)
    log.info("MODE: PREVIOUS RISK MANAGEMENT (Fixed %d shares, Pure 3:1 R:R)", qty)
    log.info("-" * 75)
    log.info("%-20s %-6s %-14s %-9s %-8s %-6s %-10s", "Timestamp", "Dir", "Rule", "Entry", "ATR", "Exit", "Net PnL")
    log.info("-" * 75)

    results = []
    total_pnl = 0.0
    wins = 0
    losses = 0
    gross_gains = 0.0
    gross_losses = 0.0

    for sig in signals:
        sig_time = pd.to_datetime(sig.timestamp, utc=True)
        # sig_time is the 15-min bar's open label; bar closes at sig_time + 15 min.
        # Must not fill on any 1-min bar inside the signal window.
        bar_close = sig_time + pd.Timedelta(minutes=15)
        future = df_1min[df_1min.index >= bar_close]
        if len(future) < 6:
            continue

        # Entry via Router
        entry_window = future.iloc[:5]
        bars_list = [row for _, row in entry_window.iterrows()]
        side = OrderSide.BUY if sig.direction == "long" else OrderSide.SELL
        actual_entry, entry_cost, filled_qty = route_order(symbol, side, qty, bars_list)
        if filled_qty == 0:
            continue

        # Scan for SL / TP
        scan = future.iloc[5:]
        highs = scan["high"].values
        lows = scan["low"].values
        exit_idx = len(scan) - 1
        raw_exit = float(scan["close"].iloc[-1])
        outcome = "open"

        for k in range(len(scan)):
            if sig.direction == "long":
                if lows[k] <= sig.stop_price:
                    raw_exit = sig.stop_price; outcome = "sl"; exit_idx = k; break
                elif highs[k] >= sig.tp_price:
                    raw_exit = sig.tp_price; outcome = "tp"; exit_idx = k; break
            else:
                if highs[k] >= sig.stop_price:
                    raw_exit = sig.stop_price; outcome = "sl"; exit_idx = k; break
                elif lows[k] <= sig.tp_price:
                    raw_exit = sig.tp_price; outcome = "tp"; exit_idx = k; break

        exit_window = scan.iloc[exit_idx: exit_idx + 5]
        if len(exit_window) == 0:
            exit_window = scan.iloc[-1:]
        out_bars = [row for _, row in exit_window.iterrows()]
        out_side = OrderSide.SELL if sig.direction == "long" else OrderSide.BUY
        actual_exit, exit_cost, _ = route_order(symbol, out_side, filled_qty, out_bars)
        if actual_exit == 0.0:
            actual_exit = raw_exit

        gross = (actual_exit - actual_entry) * filled_qty if sig.direction == "long" \
                else (actual_entry - actual_exit) * filled_qty
        net = gross - entry_cost - exit_cost
        total_pnl += net

        if net > 0:
            wins += 1
            gross_gains += net
        elif net < 0:
            losses += 1
            gross_losses += abs(net)

        results.append({
            "timestamp": sig.timestamp,
            "direction": sig.direction,
            "rule": sig.rule,
            "regime": sig.regime,
            "entry": sig.entry_price,
            "stop": sig.stop_price,
            "tp": sig.tp_price,
            "atr": sig.atr,
            "outcome": outcome,
            "net_pnl": round(net, 2),
            "running_pnl": round(total_pnl, 2),
        })

        pnl_str = f"+${net:,.2f}" if net >= 0 else f"-${abs(net):,.2f}"
        log.info(
            "%-20s %-6s %-14s $%-8.2f %-8.4f %-6s %s",
            sig.timestamp[:19],
            sig.direction.upper(),
            sig.rule[:14],
            sig.entry_price,
            sig.atr,
            outcome.upper(),
            pnl_str,
        )

    n = len(results)
    win_rate = (wins / n * 100) if n else 0.0
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else float("inf")

    log.info("=" * 65)
    log.info("SUMMARY — %s (PREVIOUS RISK MANAGEMENT)", symbol)
    log.info("  Symbol         : %s", symbol)
    log.info("  Position Size  : %d shares", qty)
    log.info("  Total Signals  : %d", n)
    log.info("  Wins / Losses  : %d / %d (Win Rate: %.1f%%)", wins, losses, win_rate)
    log.info("  Gross Gains    : +$%.2f", gross_gains)
    log.info("  Gross Losses   : -$%.2f", gross_losses)
    log.info("  Profit Factor  : %.2f", profit_factor)
    log.info("  Net Realized PnL: $%.2f", total_pnl)
    log.info("=" * 65)

    out_path = Path("results") / f"{symbol.lower()}_trades.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Trade output saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Mode 2: Dynamic Risk Management (10% DD Circuit Breaker, Trailing Stop)
# ---------------------------------------------------------------------------

def run_dynamic_risk(
    symbol: str,
    df_1min: pd.DataFrame,
    signal_map: dict[pd.Timestamp, Signal],
    capital: float = 10_000.0,
    max_dd: float = 10.0,
    risk_pct: float = 1.0,
) -> None:
    log.info("-" * 75)
    log.info("MODE: DYNAMIC RISK MANAGEMENT (Capital: $%.2f, Max DD: %.1f%%, Risk: %.1f%%)", capital, max_dd, risk_pct)
    log.info("-" * 75)

    risk_mgr = DynamicRiskManager(
        initial_capital=capital,
        max_drawdown_pct=max_dd,
        base_risk_pct=risk_pct,
    )

    open_pos: Optional[Position] = None
    pending_signal: Optional[Signal] = None  # staged signal waiting for next bar entry
    recent_bars_buffer: list[pd.Series] = []
    trade_history: list[dict] = []

    for bar_ts, row in df_1min.iterrows():
        recent_bars_buffer.append(row)
        if len(recent_bars_buffer) > 10:
            recent_bars_buffer.pop(0)

        current_price = float(row.close)
        high = float(row.high)
        low = float(row.low)

        # Mark to market & Drawdown check
        risk_mgr.mark_to_market(current_price, open_pos)

        # Circuit breaker trigger
        if risk_mgr.is_kill_switch_tripped and open_pos is not None:
            fill_price, exit_fees, _ = route_order(
                symbol, OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                open_pos.qty, recent_bars_buffer,
            )
            gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                    else (open_pos.entry_price - fill_price) * open_pos.qty
            net_pnl = gross - open_pos.entry_cost - exit_fees
            risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)
            trade_history.append({
                "entry_time": open_pos.entry_time, "exit_time": str(bar_ts),
                "direction": open_pos.direction, "rule": open_pos.rule,
                "qty": open_pos.qty, "entry_price": open_pos.entry_price, "exit_price": fill_price,
                "outcome": "KILL_SWITCH", "net_pnl": round(net_pnl, 2), "equity": round(risk_mgr.equity, 2),
                "drawdown_pct": round(risk_mgr.current_drawdown_pct, 2),
            })
            open_pos = None

        # Check active position
        if open_pos is not None and not risk_mgr.is_kill_switch_tripped:
            new_stop, _ = risk_mgr.update_stops_and_targets(open_pos, high, low, current_price)
            exit_triggered = False
            outcome = ""
            raw_exit = current_price

            if open_pos.direction == "long":
                if low <= open_pos.current_stop:
                    raw_exit = open_pos.current_stop
                    outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                    exit_triggered = True
                elif high >= open_pos.tp_price:
                    raw_exit = open_pos.tp_price; outcome = "TP"; exit_triggered = True
            else:
                if high >= open_pos.current_stop:
                    raw_exit = open_pos.current_stop
                    outcome = "TRAIL_SL" if open_pos.is_trailing_active else ("BE" if open_pos.is_breakeven_set else "SL")
                    exit_triggered = True
                elif low <= open_pos.tp_price:
                    raw_exit = open_pos.tp_price; outcome = "TP"; exit_triggered = True

            if exit_triggered:
                fill_price, exit_fees, _ = route_order(
                    symbol, OrderSide.SELL if open_pos.direction == "long" else OrderSide.BUY,
                    open_pos.qty, recent_bars_buffer,
                )
                if fill_price == 0:
                    fill_price = raw_exit
                gross = (fill_price - open_pos.entry_price) * open_pos.qty if open_pos.direction == "long" \
                        else (open_pos.entry_price - fill_price) * open_pos.qty
                net_pnl = gross - open_pos.entry_cost - exit_fees
                risk_mgr.close_position(net_pnl, open_pos.entry_cost + exit_fees)
                trade_history.append({
                    "entry_time": open_pos.entry_time, "exit_time": str(bar_ts),
                    "direction": open_pos.direction, "rule": open_pos.rule,
                    "qty": open_pos.qty, "entry_price": open_pos.entry_price, "exit_price": fill_price,
                    "outcome": outcome, "net_pnl": round(net_pnl, 2), "equity": round(risk_mgr.equity, 2),
                    "drawdown_pct": round(risk_mgr.current_drawdown_pct, 2),
                })
                open_pos = None

        # -------------------------------------------------------------------
        # Stage any signal whose 15-min bar just closed at this 1-min timestamp.
        # The signal was generated AT bar_ts (close of the 15-min window);
        # we must enter on the NEXT bar — so we only stage here, execute below.
        # -------------------------------------------------------------------
        if bar_ts in signal_map and open_pos is None and pending_signal is None:
            pending_signal = signal_map[bar_ts]

        # Execute a pending signal: we are now one bar past the signal close.
        if open_pos is None and pending_signal is not None and not risk_mgr.is_kill_switch_tripped:
            sig = pending_signal
            # Guard: only enter if this bar is strictly after the signal timestamp
            sig_ts = pd.to_datetime(sig.timestamp, utc=True)
            if bar_ts > sig_ts:
                shares = risk_mgr.calculate_position_size(sig.entry_price, sig.stop_price, sig.regime, sig.atr)
                if shares > 0:
                    fill_price, entry_fees, filled_qty = route_order(
                        symbol, OrderSide.BUY if sig.direction == "long" else OrderSide.SELL,
                        shares, recent_bars_buffer,
                    )
                    if filled_qty > 0:
                        open_pos = Position(
                            symbol=symbol, direction=sig.direction, entry_time=str(bar_ts),
                            entry_price=fill_price, qty=filled_qty, initial_stop=sig.stop_price,
                            current_stop=sig.stop_price, tp_price=sig.tp_price, atr=sig.atr,
                            rule=sig.rule, regime=sig.regime, entry_cost=entry_fees,
                        )
                pending_signal = None  # consumed regardless (avoid stale signal)

    n = len(trade_history)
    wins = sum(1 for t in trade_history if t["net_pnl"] > 0)
    losses = sum(1 for t in trade_history if t["net_pnl"] < 0)
    win_rate = (wins / n * 100) if n else 0.0
    net_pnl = risk_mgr.equity - capital
    gross_gains = sum(t["net_pnl"] for t in trade_history if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trade_history if t["net_pnl"] < 0))
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else float("inf")

    log.info("=" * 65)
    log.info("SUMMARY — %s (DYNAMIC RISK MANAGEMENT)", symbol)
    log.info("  Initial Capital       : $%.2f", capital)
    log.info("  Ending Equity         : $%.2f", risk_mgr.equity)
    log.info("  Net Realized PnL      : %s$%.2f", "+" if net_pnl >= 0 else "-", abs(net_pnl))
    log.info("  High-Water Mark (HWM) : $%.2f", risk_mgr.high_water_mark)
    log.info("  Peak Drawdown         : %.2f%% (Limit: %.1f%%)", risk_mgr.peak_drawdown_pct, max_dd)
    log.info("  Wins / Losses         : %d / %d (Win Rate: %.1f%%)", wins, losses, win_rate)
    log.info("  Profit Factor         : %.2f", profit_factor)
    log.info("  Circuit Breaker       : %s", "TRIPPED" if risk_mgr.is_kill_switch_tripped else "SAFE (0 breaches)")
    log.info("=" * 65)

    out_path = Path("results") / f"{symbol.lower()}_trades.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(trade_history).to_csv(out_path, index=False)
    log.info("Trade output saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run(
    symbol: str,
    risk_mode: str = "previous",
    qty: int = 500,
    capital: float = 10_000.0,
    max_dd: float = 10.0,
    risk_pct: float = 1.0,
    force_fetch: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> None:
    symbol = symbol.upper()
    log.info("=" * 65)
    log.info("UNIVERSAL STOCK TRADER — %s", symbol)
    log.info("Anchor Brain: AAPL + MSFT + SPY HMMs (majority vote)")
    log.info("=" * 65)

    # 1. Fetch or locate data
    csv_path = CSV_PATHS.get(symbol)
    if not csv_path:
        # Check standard Price Feed directory
        matches = list(DATA_DIR.glob(f"{symbol}_1min_*.csv"))
        if matches:
            csv_path = matches[0]
            CSV_PATHS[symbol] = csv_path

    if force_fetch or not csv_path or not Path(csv_path).exists():
        log.info("Fetching %s 1-min data from Alpaca API...", symbol)
        fetcher = AlpacaFetcher()
        fetch_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        fetch_end = datetime.now(timezone.utc)
        df_fetched = fetcher.fetch(symbol, fetch_start, fetch_end, timeframe_minutes=1)
        log.info("Fetched %d 1-min bars from Alpaca and saved to CSV", len(df_fetched))
        csv_path = CSV_PATHS.get(symbol)

    loader = DataLoader(symbol)
    df_1min = loader.load()

    if start:
        df_1min = df_1min[df_1min.index >= pd.to_datetime(start, utc=True)]
    if end:
        df_1min = df_1min[df_1min.index <= pd.to_datetime(end, utc=True)]

    log.info("Loaded %d 1-min bars (%s -> %s)", len(df_1min), df_1min.index[0].date(), df_1min.index[-1].date())

    # 2. Resample and compute features
    loader.set_dataframe(df_1min)
    df_15min = loader.resample(DEFAULT_TIMEFRAME)
    df_features = loader.compute_features(df_15min)

    # 3. Predict regimes via Anchor Brain
    models = load_anchor_models()
    regimes = majority_vote_regimes(models, df_features)
    generator = BaselineSignalGenerator(symbol)
    signals = generator.generate(df_features, regimes)
    log.info("Generated %d signals for %s", len(signals), symbol)

    if not signals:
        log.warning("No signals generated. Check date range or data coverage.")
        return

    # 4. Execute according to chosen risk management mode
    if risk_mode == "previous":
        run_previous_risk(symbol, df_1min, signals, qty=qty)
    else:
        sig_map = {pd.to_datetime(s.timestamp, utc=True): s for s in signals}
        run_dynamic_risk(symbol, df_1min, sig_map, capital=capital, max_dd=max_dd, risk_pct=risk_pct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Stock Trader using Anchor HMM Models.")
    parser.add_argument("--symbol",   type=str,   required=True, help="Stock ticker symbol (e.g. NVDA, TSLA, AMD, DIA, QQQ)")
    parser.add_argument("--risk",     type=str,   default="previous", choices=["previous", "dynamic"],
                        help="Risk management mode: 'previous' (fixed 500 shares, pure 3:1 R:R) or 'dynamic' (10%% DD cap, trailing stop)")
    parser.add_argument("--qty",      type=int,   default=500,        help="Share quantity for 'previous' mode (default 500)")
    parser.add_argument("--capital",  type=float, default=10_000.0,   help="Account capital for 'dynamic' mode (default $10,000)")
    parser.add_argument("--max-dd",   type=float, default=10.0,       help="Max drawdown %% for 'dynamic' mode (default 10.0%%)")
    parser.add_argument("--risk-pct", type=float, default=1.0,        help="Risk budget %% per trade for 'dynamic' mode (default 1.0%%)")
    parser.add_argument("--fetch",    action="store_true",            help="Force fresh Alpaca 1-min data download")
    parser.add_argument("--start",    type=str,   default=None,       help="Filter start date e.g. 2025-01-01")
    parser.add_argument("--end",      type=str,   default=None,       help="Filter end date e.g. 2026-09-27")
    args = parser.parse_args()

    run(
        symbol=args.symbol,
        risk_mode=args.risk,
        qty=args.qty,
        capital=args.capital,
        max_dd=args.max_dd,
        risk_pct=args.risk_pct,
        force_fetch=args.fetch,
        start=args.start,
        end=args.end,
    )
