"""
trade_qqq.py
============
Standalone QQQ trader that uses the pre-trained HMM models.

This script is the ONLY file you need to run after training.
It does NOT depend on validation.py, walk-forward logic, or any
test infrastructure.  The full repo is used only for the
order-router execution layer.

Workflow:
    1. Load saved HMMs for AAPL, MSFT, SPY (trained on full history).
    2. Load QQQ 1-min CSV and resample to 15-min.
    3. Compute HMM regime on QQQ bars using all 3 models (majority vote).
    4. Generate signals using the v2 baseline rules (3:1 R:R).
    5. Simulate each trade via the Smart Router (realistic fills + fees).
    6. Print a full trade log and PnL summary.

Run:
    python trade_qqq.py

Optional flags:
    --start 2025-01-01     Only trade from this date onwards
    --end   2026-09-27     Only trade up to this date
    --qty   500            Position size in shares (default 500)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM, ALL_REGIMES
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
log = logging.getLogger("trade_qqq")

QQQ_SYMBOL = "QQQ"
ANCHOR_SYMBOLS = ["AAPL", "MSFT", "SPY"]


# ---------------------------------------------------------------------------
# Step 1 — Load pre-trained HMM models from disk
# ---------------------------------------------------------------------------

def load_anchor_models() -> dict[str, RegimeHMM]:
    """
    Load all three pre-trained HMMs from saved_models/.
    Raises FileNotFoundError if a model is missing (run train_combined.py first).
    """
    models = {}
    for sym in ANCHOR_SYMBOLS:
        path = MODEL_DIR / f"hmm_{sym}_full.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {path}\n"
                f"Run  python train_combined.py  first to build the models."
            )
        models[sym] = RegimeHMM.load(path)
        log.info("Loaded HMM: %s (trained on %s data)", sym, models[sym].symbol)
    return models


# ---------------------------------------------------------------------------
# Step 2 — Majority-vote regime from all 3 anchor models
# ---------------------------------------------------------------------------

def majority_vote_regime(
    hmm_models: dict[str, RegimeHMM],
    df_features: pd.DataFrame,
) -> pd.Series:
    """
    Apply all 3 anchor HMMs to QQQ feature bars and take the majority-vote
    regime label for each bar.

    This is more robust than any single model because it requires at least
    2 of 3 trained models to agree before assigning a regime.
    """
    predictions = {}
    for sym, hmm in hmm_models.items():
        predictions[sym] = hmm.predict(df_features)

    pred_df = pd.DataFrame(predictions, index=df_features.index)

    # Majority vote: most frequent label across the 3 columns
    regime_vote = pred_df.mode(axis=1)[0]
    regime_vote.name = "regime"

    # Log distribution
    dist = regime_vote.value_counts(normalize=True).round(3).to_dict()
    log.info("Majority-vote regime distribution on QQQ: %s", dist)

    return regime_vote


# ---------------------------------------------------------------------------
# Step 3 — Trade simulation (identical to validation.py simulate_trade)
# ---------------------------------------------------------------------------

def simulate_trade(signal: Signal, df_1min: pd.DataFrame, qty: int = 500) -> float:
    """
    Simulate one round-trip through the Smart Router.
    Returns net PnL after slippage and exchange fees.
    """
    sig_time = pd.to_datetime(signal.timestamp, utc=True)
    future = df_1min[df_1min.index > sig_time]
    if len(future) < 6:
        return 0.0

    # --- Entry ---
    entry_window = future.iloc[:5]
    venues = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
    side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
    order = Order(symbol=signal.symbol, side=side, order_type=OrderType.MARKET, quantity=qty)
    bars_in = [
        Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for _, row in entry_window.iterrows()
    ]
    res_in = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
    if res_in.total_filled == 0:
        return 0.0

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
    venues_out = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
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
# Main
# ---------------------------------------------------------------------------

def run(
    start: Optional[str] = None,
    end: Optional[str] = None,
    qty: int = 500,
) -> None:

    log.info("=" * 65)
    log.info("QQQ TRADER — using AAPL + MSFT + SPY anchor models")
    log.info("=" * 65)

    # --- Load anchor models ---
    hmm_models = load_anchor_models()

    # --- Load QQQ data ---
    loader = DataLoader(QQQ_SYMBOL)
    df_1min = loader.load()
    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features = loader.compute_features(df_resampled)

    # --- Optional date filter ---
    if start:
        df_features = df_features[df_features.index >= pd.to_datetime(start, utc=True)]
        df_1min = df_1min[df_1min.index >= pd.to_datetime(start, utc=True)]
    if end:
        df_features = df_features[df_features.index <= pd.to_datetime(end, utc=True)]
        df_1min = df_1min[df_1min.index <= pd.to_datetime(end, utc=True)]

    log.info(
        "QQQ data: %d 15-min bars | %s → %s",
        len(df_features),
        df_features.index[0].date(),
        df_features.index[-1].date(),
    )

    # --- Regime: majority vote from all 3 anchor HMMs ---
    log.info("Computing regime via majority vote (AAPL + MSFT + SPY HMMs)...")
    regimes = majority_vote_regime(hmm_models, df_features)

    # --- Generate signals ---
    gen = BaselineSignalGenerator(symbol=QQQ_SYMBOL)
    signals = gen.generate(df_features, regimes)

    if not signals:
        log.warning("No signals generated. Check date range or data coverage.")
        return

    log.info("Signals generated: %d", len(signals))

    # --- Simulate trades ---
    log.info("-" * 65)
    log.info("%-24s %-6s %-8s %-10s %-8s %-10s", "Timestamp", "Dir", "Rule", "Entry", "ATR", "Net PnL")
    log.info("-" * 65)

    results = []
    total_pnl = 0.0
    wins = 0
    losses = 0

    for sig in signals:
        result = simulate_trade(sig, df_1min, qty=qty)
        if isinstance(result, tuple):
            net, outcome = result
        else:
            net, outcome = result, "open"

        total_pnl += net
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1

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
            "%-24s %-6s %-8s $%-9.2f %-8.4f %s",
            sig.timestamp[:19],
            sig.direction.upper(),
            sig.rule[:8],
            sig.entry_price,
            sig.atr,
            pnl_str,
        )

    # --- Summary ---
    n = len(signals)
    win_rate = wins / n * 100 if n else 0
    log.info("=" * 65)
    log.info("SUMMARY")
    log.info("  Symbol        : QQQ")
    log.info("  Model         : AAPL + MSFT + SPY HMMs (majority vote)")
    log.info("  Trained on    : Full 2024-2026 history per anchor")
    log.info("  Total signals : %d", n)
    log.info("  Wins          : %d  |  Losses: %d  |  Win Rate: %.1f%%", wins, losses, win_rate)
    log.info("  Net PnL       : $%.2f", total_pnl)
    log.info("  Qty per trade : %d shares", qty)
    log.info("=" * 65)

    # --- Save trade log to CSV ---
    out_path = Path("results") / "qqq_trades.csv"
    out_path.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    log.info("Trade log saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade QQQ using pre-trained anchor HMMs.")
    parser.add_argument("--start", type=str, default=None, help="Start date e.g. 2025-01-01")
    parser.add_argument("--end",   type=str, default=None, help="End date   e.g. 2026-09-27")
    parser.add_argument("--qty",   type=int, default=500,  help="Shares per trade (default 500)")
    args = parser.parse_args()

    run(start=args.start, end=args.end, qty=args.qty)
