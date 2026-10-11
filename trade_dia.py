"""
trade_dia.py
============
Standalone Dow Jones (DIA — SPDR Dow Jones Industrial Average ETF) trader
using a DIA-specific self-trained HMM for regime classification.

Each ticker should use its own HMM trained on its own data.
Cross-market majority voting from AAPL/MSFT/SPY caused regime mis-labelling
and degraded performance.  This script uses DIA's own HMM.

Usage:
    # Run backtest on local DIA data (fetched from Alpaca)
    python trade_dia.py

    # Fetch latest data from Alpaca before trading
    python trade_dia.py --fetch

    # Custom date filter and position size
    python trade_dia.py --start 2025-01-01 --end 2026-09-27 --qty 500
"""

from __future__ import annotations

import argparse
import logging
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

# Order Router for realistic execution costs
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import SmartStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("trade_dia")

DIA_SYMBOL = "DIA"


# ---------------------------------------------------------------------------
# Step 1 — Load DIA's own self-trained HMM
# ---------------------------------------------------------------------------

def load_dia_model() -> RegimeHMM:
    """
    Load DIA's own HMM from saved_models/.
    Falls back to SPY (closest correlated index) if DIA model is not found.
    Run train_combined.py to generate the DIA-specific model.
    """
    # Primary: DIA's own self-trained HMM
    dia_path = MODEL_DIR / "hmm_DIA_full.pkl"
    if dia_path.exists():
        model = RegimeHMM.load(dia_path)
        log.info("Loaded DIA self-trained HMM (best regime accuracy)")
        return model

    # Fallback: SPY (high correlation with DIA, both large-cap US equities)
    spy_path = MODEL_DIR / "hmm_SPY_full.pkl"
    if spy_path.exists():
        log.warning(
            "DIA self-model not found at %s. "
            "Falling back to SPY HMM (correlated large-cap US index). "
            "Run 'python train_combined.py' to generate the DIA-specific model.",
            dia_path,
        )
        return RegimeHMM.load(spy_path)

    raise FileNotFoundError(
        f"No HMM model found for DIA.\n"
        f"Run  python train_combined.py  first to build the models."
    )


# ---------------------------------------------------------------------------
# Step 3 — Trade simulation through Smart Router
# ---------------------------------------------------------------------------

def simulate_trade(signal: Signal, df_1min: pd.DataFrame, qty: int = 500) -> tuple[float, str]:
    """
    Simulate one round-trip through the Smart Router.
    Returns (net_pnl, outcome).
    """
    sig_time = pd.to_datetime(signal.timestamp, utc=True)
    # Entry must be at or after the bar's close (sig_time is the bar open label;
    # a 15-min bar closes at sig_time + 15 min).  Bars strictly inside the signal
    # window must not be used — this is the lookahead guard.
    bar_close = sig_time + pd.Timedelta(minutes=15)
    future = df_1min[df_1min.index >= bar_close]
    if len(future) < 6:
        return 0.0, "insufficient_data"

    # --- Entry ---
    entry_window = future.iloc[:5]
    venues = {name: Exchange(name, cfg, symbol=signal.symbol) for name, cfg in VENUE_CONFIGS.items()}
    side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
    order = Order(symbol=signal.symbol, side=side, order_type=OrderType.MARKET, quantity=qty)
    bars_in = [
        Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for _, row in entry_window.iterrows()
    ]
    res_in = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
    if res_in.total_filled == 0:
        return 0.0, "unfilled_entry"

    actual_entry = res_in.avg_fill_price
    entry_cost = res_in.total_fees
    filled_qty = res_in.total_filled

    # --- Scan for SL / TP ---
    scan = future.iloc[5:]
    highs = scan["high"].values
    lows = scan["low"].values
    exit_idx = len(scan) - 1
    raw_exit = float(scan["close"].iloc[-1])
    outcome = "open"

    for k in range(len(scan)):
        if signal.direction == "long":
            if lows[k] <= signal.stop_price:
                raw_exit = signal.stop_price; outcome = "sl"; exit_idx = k; break
            elif highs[k] >= signal.tp_price:
                raw_exit = signal.tp_price;  outcome = "tp"; exit_idx = k; break
        else:
            if highs[k] >= signal.stop_price:
                raw_exit = signal.stop_price; outcome = "sl"; exit_idx = k; break
            elif lows[k] <= signal.tp_price:
                raw_exit = signal.tp_price;  outcome = "tp"; exit_idx = k; break

    # --- Exit ---
    exit_window = scan.iloc[exit_idx: exit_idx + 5]
    if len(exit_window) == 0:
        exit_window = scan.iloc[-1:]
    venues_out = {name: Exchange(name, cfg, symbol=signal.symbol) for name, cfg in VENUE_CONFIGS.items()}
    out_side = OrderSide.SELL if signal.direction == "long" else OrderSide.BUY
    out_order = Order(symbol=signal.symbol, side=out_side, order_type=OrderType.MARKET, quantity=filled_qty)
    bars_out = [
        Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for _, row in exit_window.iterrows()
    ]
    res_out = OrderRouter(venues=venues_out, strategy=SmartStrategy()).submit(out_order, bars_out)
    actual_exit = res_out.avg_fill_price if res_out.total_filled > 0 else raw_exit
    exit_cost = res_out.total_fees if res_out.total_filled > 0 else 0.0

    gross = (actual_exit - actual_entry) * filled_qty if signal.direction == "long" \
            else (actual_entry - actual_exit) * filled_qty

    net = gross - entry_cost - exit_cost
    return net, outcome


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def run(
    do_fetch: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    qty: int = 500,
) -> None:

    log.info("=" * 65)
    log.info("DOW JONES (DIA) TRADER — self-trained HMM (no cross-market vote)")
    log.info("=" * 65)

    # 1. Fetch from Alpaca if requested or missing
    csv_path = CSV_PATHS.get(DIA_SYMBOL)
    if do_fetch or csv_path is None or not Path(csv_path).exists():
        log.info("Fetching DIA 1-min data from Alpaca API...")
        fetcher = AlpacaFetcher()
        fetch_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        fetch_end = datetime.now(timezone.utc)
        df_fetched = fetcher.fetch(DIA_SYMBOL, fetch_start, fetch_end, timeframe_minutes=1)
        log.info("Fetched %d 1-min bars from Alpaca and saved to CSV", len(df_fetched))

    # 2. Load DIA's own HMM
    hmm = load_dia_model()

    # 3. Load DIA data
    loader = DataLoader(DIA_SYMBOL)
    df_1min = loader.load()
    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features = loader.compute_features(df_resampled)

    # 4. Optional date filtering
    if start:
        df_features = df_features[df_features.index >= pd.to_datetime(start, utc=True)]
        df_1min = df_1min[df_1min.index >= pd.to_datetime(start, utc=True)]
    if end:
        df_features = df_features[df_features.index <= pd.to_datetime(end, utc=True)]
        df_1min = df_1min[df_1min.index <= pd.to_datetime(end, utc=True)]

    log.info(
        "DIA data: %d 15-min bars | %s → %s",
        len(df_features),
        df_features.index[0].date(),
        df_features.index[-1].date(),
    )

    # 5. Self-regime classification
    log.info("Classifying DIA regime with DIA-specific HMM...")
    regimes = hmm.predict(df_features)
    dist = regimes.value_counts(normalize=True).round(3).to_dict()
    log.info("DIA regime distribution: %s", dist)

    # 6. Generate signals — DIA-specific thresholds (lower body/vol for low-vol ETF)
    gen = BaselineSignalGenerator(
        symbol=DIA_SYMBOL,
        body_confirm=0.40,   # DIA has smaller candle bodies than QQQ
        vol_confirm=1.05,    # Slightly lower vol bar required
    )
    signals = gen.generate(df_features, regimes)

    if not signals:
        log.warning("No signals generated. Check date range or data coverage.")
        return

    log.info("Generated %d signals for DIA", len(signals))

    # 7. Simulate trades
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
        net, outcome = simulate_trade(sig, df_1min, qty=qty)

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

    # 8. Summary statistics
    n = len(signals)
    win_rate = (wins / n * 100) if n else 0.0
    profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else float("inf")

    log.info("=" * 65)
    log.info("SUMMARY — DOW JONES (DIA)")
    log.info("  Symbol         : DIA (SPDR Dow Jones Industrial Average ETF)")
    log.info("  Source         : Alpaca 1-min Data")
    log.info("  Regime model   : DIA self-trained HMM")
    log.info("  Total signals  : %d", n)
    log.info("  Wins           : %d", wins)
    log.info("  Losses         : %d", losses)
    log.info("  Win Rate       : %.1f%%", win_rate)
    log.info("  Gross Gains    : +$%.2f", gross_gains)
    log.info("  Gross Losses   : -$%.2f", gross_losses)
    log.info("  Profit Factor  : %.2f", profit_factor)
    log.info("  Net PnL        : $%.2f", total_pnl)
    log.info("  Shares / Trade : %d", qty)
    log.info("=" * 65)

    # 9. Save trade log
    out_path = Path("results") / "dia_trades.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Trade log saved -> %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Dow Jones (DIA) from Alpaca 1-min data using anchor HMMs.")
    parser.add_argument("--fetch", action="store_true", help="Force fresh fetch from Alpaca")
    parser.add_argument("--start", type=str, default=None, help="Start date e.g. 2025-01-01")
    parser.add_argument("--end",   type=str, default=None, help="End date e.g. 2026-09-27")
    parser.add_argument("--qty",   type=int, default=500,  help="Shares per trade (default 500)")
    args = parser.parse_args()

    run(do_fetch=args.fetch, start=args.start, end=args.end, qty=args.qty)
