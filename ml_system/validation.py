"""
Validation — Expanding window walk-forward validation for Phase 1 LSTM filter.
Tests all three equity symbols (AAPL, MSFT, SPY) sequentially.

Key design:
- 1-min CSV data is converted to 15-min bars in streaming chunks to simulate
  live market feed behaviour (no look-ahead beyond the current 15-min bar).
- Signals routed through OrderRouter SmartStrategy for real execution costs.
- Per-fold DSR + SPA gate logged to registry.
"""

import logging
import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ml_system.config import (
    DEFAULT_TIMEFRAME,
    EQUITY_SYMBOLS,
)
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.models.lstm_filter import SignalFilterLSTM
from ml_system.signals.baseline_rules import BaselineSignalGenerator, Signal
from ml_system.stats import deflated_sharpe_ratio, block_bootstrap_spa
from ml_system.registry.retrain_log import RetrainLog

# Order Router for realistic execution costs
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import SmartStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
log = logging.getLogger("validation")


# ---------------------------------------------------------------------------
# Execution Simulator
# ---------------------------------------------------------------------------

def simulate_trade(signal: Signal, df_1min: pd.DataFrame) -> float:
    """
    Simulate one round-trip trade through the Smart Router.
    Returns net PnL after execution costs.
    Streams 1-min bars in 5-bar entry window, then scans for SL/TP hit.
    """
    QTY = 500

    sig_time = pd.to_datetime(signal.timestamp, utc=True)
    future = df_1min[df_1min.index > sig_time]
    if len(future) < 6:
        return 0.0

    # --- Entry: execute over first 5 1-min bars ---
    entry_window = future.iloc[:5]
    venues = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
    side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
    order = Order(symbol=signal.symbol, side=side, order_type=OrderType.MARKET, quantity=QTY)
    bars_in = [
        Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for _, row in entry_window.iterrows()
    ]
    res_in = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
    if res_in.total_filled == 0:
        return 0.0

    actual_entry = res_in.avg_fill_price
    entry_cost   = res_in.total_fees
    filled_qty   = res_in.total_filled

    # --- Scan for SL/TP hit on subsequent 1-min bars ---
    scan = future.iloc[5:]
    highs  = scan["high"].values
    lows   = scan["low"].values
    outcome = "open"
    exit_idx = len(scan) - 1
    raw_exit = float(scan["close"].iloc[-1])

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

    # --- Exit: execute over next 5 1-min bars after trigger ---
    exit_window = scan.iloc[exit_idx: exit_idx + 5]
    if len(exit_window) == 0:
        exit_window = scan.iloc[-1:]
    venues_out  = {name: Exchange(name, cfg) for name, cfg in VENUE_CONFIGS.items()}
    out_side    = OrderSide.SELL if signal.direction == "long" else OrderSide.BUY
    out_order   = Order(symbol=signal.symbol, side=out_side, order_type=OrderType.MARKET, quantity=filled_qty)
    bars_out    = [
        Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
            row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
        for _, row in exit_window.iterrows()
    ]
    res_out     = OrderRouter(venues=venues_out, strategy=SmartStrategy()).submit(out_order, bars_out)
    actual_exit = res_out.avg_fill_price if res_out.total_filled > 0 else raw_exit
    exit_cost   = res_out.total_fees if res_out.total_filled > 0 else 0.0

    if signal.direction == "long":
        gross = (actual_exit - actual_entry) * filled_qty
    else:
        gross = (actual_entry - actual_exit) * filled_qty

    return gross - entry_cost - exit_cost


def evaluate_signals(signals: List[Signal], df_1min: pd.DataFrame) -> pd.Series:
    """Evaluate signals and return a daily PnL series."""
    if not signals:
        return pd.Series(dtype=float)
    pnls  = [simulate_trade(s, df_1min) for s in signals]
    dates = [pd.to_datetime(s.timestamp) for s in signals]
    df_r  = pd.DataFrame({"pnl": pnls}, index=dates)
    return df_r.resample("1D").sum().fillna(0)["pnl"]


def _run_fold_stats(
    fold: int,
    symbol: str,
    sig_base: List[Signal],
    sig_lstm: List[Signal],
    df_1min_oos: pd.DataFrame,
    use_lstm: bool,
    n_trials: int,
) -> dict:
    """
    Shared helper: evaluate signals, compute DSR+SPA stats, log results,
    and return a fold-result dict.  Used by all three walk-forward variants.
    """
    daily_base = evaluate_signals(sig_base, df_1min_oos)
    daily_lstm = evaluate_signals(sig_lstm, df_1min_oos) if use_lstm else daily_base

    df_ret = pd.concat([daily_lstm.rename("lstm"), daily_base.rename("base")], axis=1).fillna(0.0)

    # --- Statistics ---
    # periods_per_year = 252*26 (equity 15-min bars; corrected from 252*24*4)
    dsr   = deflated_sharpe_ratio(df_ret["lstm"], expected_sr=0.0, n_trials=n_trials)
    spa_p = block_bootstrap_spa(df_ret["lstm"], df_ret["base"])

    base_pnl      = df_ret["base"].sum()
    lstm_pnl      = df_ret["lstm"].sum()
    n_trades_base = len(sig_base)
    n_wins_base   = int((daily_base > 0).sum())
    win_rate      = round(n_wins_base / n_trades_base * 100, 1) if n_trades_base else 0.0
    promoted      = (dsr > 0.95 and spa_p < 0.05) if use_lstm else None

    log.info("  Phase 0 PnL: $%.2f  |  Daily win days: %d/%d (%.1f%%)",
             base_pnl, n_wins_base, len(daily_base[daily_base != 0]), win_rate)
    if use_lstm:
        log.info("  Phase 1 PnL: $%.2f  |  DSR: %.3f  |  SPA p: %.4f  |  Gate: %s",
                 lstm_pnl, dsr, spa_p, "PASSED" if promoted else "FAILED")

    return {
        "symbol":    symbol,
        "fold":      fold,
        "n_signals": n_trades_base,
        "base_pnl":  round(base_pnl, 2),
        "lstm_pnl":  round(lstm_pnl, 2) if use_lstm else None,
        "dsr":       round(dsr, 4)      if use_lstm else None,
        "spa_p":     round(spa_p, 4)    if use_lstm else None,
        "promoted":  promoted,
    }


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------

def run_walk_forward(symbol: str, n_folds: int = 3, use_lstm: bool = False) -> Dict:
    """
    Expanding-window walk-forward validation for one symbol.
    Streams 1-min data → resamples to 15-min inside each fold.
    Returns a summary dict.
    """
    log.info("=" * 60)
    log.info("Walk-Forward: %s  (%d folds, LSTM=%s)", symbol, n_folds, use_lstm)
    log.info("=" * 60)

    loader  = DataLoader(symbol)
    df_1min = loader.load()              # raw 1-min cache

    # Full 15-min features (for fold slicing)
    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features  = loader.compute_features(df_resampled)

    total_bars = len(df_features)
    min_train  = min(6000, int(total_bars * 0.4))
    oos_size   = max(100, (total_bars - min_train) // n_folds)

    registry = RetrainLog()
    fold_results = []

    for fold in range(n_folds):
        train_end = min_train + fold * oos_size
        oos_end   = min(train_end + oos_size, total_bars)

        df_train = df_features.iloc[:train_end]
        df_oos   = df_features.iloc[train_end:oos_end]

        if len(df_oos) < 50:
            log.warning("Fold %d: not enough OOS bars, skipping.", fold + 1)
            continue

        log.info("--- Fold %d/%d  |  Train %s→%s (%d bars)  |  OOS %s→%s (%d bars)",
                 fold + 1, n_folds,
                 df_train.index[0].date(), df_train.index[-1].date(), len(df_train),
                 df_oos.index[0].date(), df_oos.index[-1].date(), len(df_oos))

        # 1. HMM
        hmm = RegimeHMM(symbol=symbol)
        hmm.fit(df_train)
        regimes_oos = hmm.predict(df_oos)

        # 2. (Optional) LSTM
        lstm_probs = None
        if use_lstm:
            regimes_train = hmm.predict(df_train)
            lstm_features_train = loader.compute_lstm_features(df_train, regimes_train)
            lstm = SignalFilterLSTM(symbol)
            lstm.fit(lstm_features_train)
            lstm_features_oos = loader.compute_lstm_features(df_oos, regimes_oos)
            lstm_probs = lstm.predict(lstm_features_oos)

        # 3. Signals — baseline vs filtered (15-min streaming simulation)
        gen = BaselineSignalGenerator(symbol=symbol)
        sig_base = gen.generate(df_oos, regimes_oos)
        sig_lstm = gen.generate(df_oos, regimes_oos, lstm_probs=lstm_probs) if use_lstm else sig_base

        log.info("  Phase 0 signals: %d  |  Phase 1 (LSTM-filtered): %d", len(sig_base), len(sig_lstm))

        # 4. Simulate PnL through Smart Router using 1-min data
        oos_start  = df_oos.index[0]
        oos_end_ts = df_oos.index[-1]
        df_1min_oos = df_1min[(df_1min.index >= oos_start) & (df_1min.index <= oos_end_ts)]

        result = _run_fold_stats(
            fold=fold + 1, symbol=symbol,
            sig_base=sig_base, sig_lstm=sig_lstm,
            df_1min_oos=df_1min_oos,
            use_lstm=use_lstm, n_trials=fold + 1,
        )

        # 5. Registry
        registry.log_training_run(
            symbol=symbol,
            train_start=str(df_train.index[0]),
            train_end=str(df_train.index[-1]),
            train_bars=len(df_train),
            regime_distribution=hmm.regime_distribution(df_train),
            feature_set_hash=f"baseline_v2_fold{fold+1}",
            log_likelihood=0.0,
            n_states=4,
            model_path="",
        )

        fold_results.append(result)

    return fold_results


def run_alternating_walk_forward(symbol: str, use_lstm: bool = False) -> List[Dict]:
    """
    Alternating-block walk-forward validation for one symbol.
    Slices the dataset into 10% blocks.
    Strictly alternates: 1 block Train -> 1 block OOS -> next block Train ...
    Produces 5 folds (20% of data per fold: 10% train, 10% test).
    """
    log.info("=" * 60)
    log.info("Alternating Walk-Forward: %s  (10%% Train / 10%% OOS blocks, LSTM=%s)", symbol, use_lstm)
    log.info("=" * 60)

    loader  = DataLoader(symbol)
    df_1min = loader.load()              # raw 1-min cache

    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features  = loader.compute_features(df_resampled)

    total_bars = len(df_features)
    block_size = int(total_bars * 0.10)
    n_folds = 5

    registry = RetrainLog()
    fold_results = []

    for fold in range(n_folds):
        train_start = (fold * 2) * block_size
        train_end   = train_start + block_size
        oos_start   = train_end
        oos_end     = min(oos_start + block_size, total_bars)

        df_train = df_features.iloc[train_start:train_end]
        df_oos   = df_features.iloc[oos_start:oos_end]

        if len(df_oos) < 50 or len(df_train) < 50:
            log.warning("Fold %d: not enough bars, skipping.", fold + 1)
            continue

        log.info("--- Fold %d/%d  |  Train %s→%s (%d bars)  |  OOS %s→%s (%d bars)",
                 fold + 1, n_folds,
                 df_train.index[0].date(), df_train.index[-1].date(), len(df_train),
                 df_oos.index[0].date(), df_oos.index[-1].date(), len(df_oos))

        # 1. HMM
        hmm = RegimeHMM(symbol=symbol)
        hmm.fit(df_train)
        regimes_oos = hmm.predict(df_oos)

        # 2. (Optional) LSTM
        lstm_probs = None
        if use_lstm:
            regimes_train = hmm.predict(df_train)
            lstm_features_train = loader.compute_lstm_features(df_train, regimes_train)
            lstm = SignalFilterLSTM(symbol)
            lstm.fit(lstm_features_train)
            lstm_features_oos = loader.compute_lstm_features(df_oos, regimes_oos)
            lstm_probs = lstm.predict(lstm_features_oos)

        # 3. Signals — baseline vs filtered (15-min streaming simulation)
        gen = BaselineSignalGenerator(symbol=symbol)
        sig_base = gen.generate(df_oos, regimes_oos)
        sig_lstm = gen.generate(df_oos, regimes_oos, lstm_probs=lstm_probs) if use_lstm else sig_base

        log.info("  Phase 0 signals: %d  |  Phase 1 (LSTM-filtered): %d", len(sig_base), len(sig_lstm))

        # 4. Simulate PnL through Smart Router using 1-min data
        oos_start_ts = df_oos.index[0]
        oos_end_ts   = df_oos.index[-1]
        df_1min_oos  = df_1min[(df_1min.index >= oos_start_ts) & (df_1min.index <= oos_end_ts)]

        result = _run_fold_stats(
            fold=fold + 1, symbol=symbol,
            sig_base=sig_base, sig_lstm=sig_lstm,
            df_1min_oos=df_1min_oos,
            use_lstm=use_lstm, n_trials=fold + 1,
        )

        # 5. Registry
        registry.log_training_run(
            symbol=symbol,
            train_start=str(df_train.index[0]),
            train_end=str(df_train.index[-1]),
            train_bars=len(df_train),
            regime_distribution=hmm.regime_distribution(df_train),
            feature_set_hash=f"baseline_v2_alt_fold{fold+1}",
            log_likelihood=0.0,
            n_states=4,
            model_path="",
        )

        fold_results.append(result)

    return fold_results


def extract_fold_windows(anchor_symbol: str = "AAPL", n_folds: int = 3) -> List[Dict]:
    """
    Extract the exact calendar timestamps for the train/OOS splits from an anchor symbol.
    """
    loader = DataLoader(anchor_symbol)
    df_1min = loader.load()
    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features  = loader.compute_features(df_resampled)

    total_bars = len(df_features)
    min_train  = min(6000, int(total_bars * 0.4))
    oos_size   = max(100, (total_bars - min_train) // n_folds)

    windows = []
    for fold in range(n_folds):
        train_end = min_train + fold * oos_size
        oos_end   = min(train_end + oos_size, total_bars)

        t_train_start = df_features.index[0]
        t_train_end   = df_features.index[train_end - 1] if train_end > 0 else df_features.index[0]
        t_oos_start   = df_features.index[train_end] if train_end < total_bars else df_features.index[-1]
        t_oos_end     = df_features.index[oos_end - 1] if oos_end > 0 else df_features.index[-1]
        
        windows.append({
            "fold": fold + 1,
            "train_start": t_train_start,
            "train_end": t_train_end,
            "oos_start": t_oos_start,
            "oos_end": t_oos_end
        })
    return windows

def run_fixed_folds(target_symbol: str, fold_windows: List[Dict], anchor_symbol: str = "AAPL", use_lstm: bool = False) -> List[Dict]:
    """
    Test a target symbol purely Out-Of-Sample, using regime models trained on an anchor symbol
    over the exact same calendar windows.
    """
    log.info("=" * 60)
    log.info("Fixed-Fold Transfer: %s (Anchor: %s, LSTM=%s)", target_symbol, anchor_symbol, use_lstm)
    log.info("=" * 60)

    anchor_loader = DataLoader(anchor_symbol)
    anchor_1min = anchor_loader.load()
    anchor_features = anchor_loader.compute_features(anchor_loader.resample(DEFAULT_TIMEFRAME))

    target_loader = DataLoader(target_symbol)
    
    # Check if target needs fetching
    if target_symbol not in ["AAPL", "MSFT", "SPY"]:
        from ml_system.data.alpaca_fetcher import AlpacaFetcher
        fetcher = AlpacaFetcher()
        anchor_start = anchor_1min.index[0].to_pydatetime()
        anchor_end = anchor_1min.index[-1].to_pydatetime()
        df_target_raw = fetcher.fetch_if_missing(target_symbol, start=anchor_start, end=anchor_end, timeframe_minutes=1 if target_symbol != "GC" else 60)
        target_loader.set_dataframe(df_target_raw)

    target_1min = target_loader.load() if target_loader._df_1min is None else target_loader._df_1min
    target_features = target_loader.compute_features(target_loader.resample(DEFAULT_TIMEFRAME))

    fold_results = []

    for w in fold_windows:
        fold = w["fold"]
        
        df_train_anchor = anchor_features[(anchor_features.index >= w["train_start"]) & (anchor_features.index <= w["train_end"])]
        df_oos_target = target_features[(target_features.index >= w["oos_start"]) & (target_features.index <= w["oos_end"])]

        if len(df_oos_target) < 50:
            log.warning("Fold %d: Target %s not enough OOS bars, skipping.", fold, target_symbol)
            continue

        log.info("--- Fold %d/%d  |  Anchor Train %s→%s (%d bars)  |  Target OOS %s→%s (%d bars)",
                 fold, len(fold_windows),
                 df_train_anchor.index[0].date(), df_train_anchor.index[-1].date(), len(df_train_anchor),
                 df_oos_target.index[0].date(), df_oos_target.index[-1].date(), len(df_oos_target))

        hmm = RegimeHMM(symbol=anchor_symbol)
        hmm.fit(df_train_anchor)
        regimes_oos = hmm.predict(df_oos_target)

        lstm_probs = None
        if use_lstm:
            regimes_train_anchor = hmm.predict(df_train_anchor)
            lstm_features_train = anchor_loader.compute_lstm_features(df_train_anchor, regimes_train_anchor)
            lstm = SignalFilterLSTM(anchor_symbol)
            lstm.fit(lstm_features_train)
            
            lstm_features_oos = target_loader.compute_lstm_features(df_oos_target, regimes_oos)
            lstm_probs = lstm.predict(lstm_features_oos)

        gen = BaselineSignalGenerator(symbol=target_symbol)
        sig_base = gen.generate(df_oos_target, regimes_oos)
        sig_lstm = gen.generate(df_oos_target, regimes_oos, lstm_probs=lstm_probs) if use_lstm else sig_base

        log.info("  Phase 0 signals: %d  |  Phase 1 (LSTM-filtered): %d", len(sig_base), len(sig_lstm))

        oos_start = df_oos_target.index[0]
        oos_end_ts = df_oos_target.index[-1]
        df_1min_oos = target_1min[(target_1min.index >= oos_start) & (target_1min.index <= oos_end_ts)]

        result = _run_fold_stats(
            fold=fold, symbol=target_symbol,
            sig_base=sig_base, sig_lstm=sig_lstm,
            df_1min_oos=df_1min_oos,
            use_lstm=use_lstm, n_trials=fold,
        )
        fold_results.append(result)

    return fold_results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "SPY"])
    parser.add_argument("--folds",   type=int, default=3)
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--fixed-folds", action="store_true", help="Run in OOS transfer mode against AAPL anchor windows")
    parser.add_argument("--alternating", action="store_true", help="Run alternating block validation (10% train, 10% OOS)")
    args = parser.parse_args()


    all_results = []
    if args.fixed_folds:
        log.info("Extracting anchor fold windows from AAPL...")
        fold_windows = extract_fold_windows("AAPL", n_folds=args.folds)
        for sym in args.symbols:
            results = run_fixed_folds(sym, fold_windows, anchor_symbol="AAPL", use_lstm=args.use_lstm)
            all_results.extend(results)
    elif args.alternating:
        for sym in args.symbols:
            results = run_alternating_walk_forward(sym, use_lstm=args.use_lstm)
            all_results.extend(results)
    else:
        for sym in args.symbols:
            results = run_walk_forward(sym, n_folds=args.folds, use_lstm=args.use_lstm)
            all_results.extend(results)

    # --- Summary Table ---
    log.info("\n" + "=" * 70)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 70)
    log.info("%-8s %-6s %-10s %-14s %-14s", "Symbol", "Fold", "Signals", "Phase0 PnL", "Phase1 PnL")
    log.info("-" * 70)
    total_base = 0.0
    total_lstm = 0.0
    for r in all_results:
        lstm_col = f"${r['lstm_pnl']:>10.2f}" if r["lstm_pnl"] is not None else "         n/a"
        log.info("%-8s %-6d %-10d $%-13.2f %s",
                 r["symbol"], r["fold"], r["n_signals"], r["base_pnl"], lstm_col)
        total_base += r["base_pnl"]
        if r["lstm_pnl"]:
            total_lstm += r["lstm_pnl"]

    log.info("-" * 70)
    log.info("%-8s %-6s %-10s $%-13.2f $%-13.2f", "TOTAL", "", "", total_base, total_lstm)
    log.info("=" * 70)
