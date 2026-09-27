"""
trade_gc.py
===========
Standalone Gold Futures (GC=F) trader using the pre-trained anchor HMMs.

Uses the GC_15min_2026.csv (Jul–Sep 2026, yfinance) as the trading data.
Regime is determined by majority vote across AAPL + MSFT + SPY full-history models.

Key differences vs trade_qqq.py:
  - GC trades nearly 24h (no equity 9:45-15:15 ET filter applied)
  - GC uses pip-based cost model, not bps
  - 15-min bars are used for both signal generation AND exit scanning
    (no separate 1-min feed available for GC)
  - Position size is in contracts/units, not equity shares
    (default 10 contracts ≈ comparable notional to QQQ 500 shares)

Run:
    python trade_gc.py
    python trade_gc.py --start 2026-08-01 --end 2026-09-25 --qty 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR, CSV_PATHS
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator, Signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("trade_gc")

GC_SYMBOL = "GC"
ANCHOR_SYMBOLS = ["AAPL", "MSFT", "SPY"]

# GC cost model: $0.10/contract/side spread + $0.50/contract commission
GC_SPREAD_PER_UNIT = 0.50     # approx 0.5 pip spread on GC futures
GC_COMMISSION      = 0.10     # per unit per side
GC_SLIPPAGE        = 0.30     # slippage per unit per side


# ---------------------------------------------------------------------------
# Load pre-trained anchor models
# ---------------------------------------------------------------------------

def load_anchor_models() -> dict[str, RegimeHMM]:
    models = {}
    for sym in ANCHOR_SYMBOLS:
        path = MODEL_DIR / f"hmm_{sym}_full.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run  python train_combined.py  first."
            )
        models[sym] = RegimeHMM.load(path)
        log.info("Loaded HMM: %s", sym)
    return models


# ---------------------------------------------------------------------------
# Majority-vote regime
# ---------------------------------------------------------------------------

def majority_vote_regime(
    hmm_models: dict[str, RegimeHMM],
    df_features: pd.DataFrame,
) -> pd.Series:
    predictions = {sym: hmm.predict(df_features) for sym, hmm in hmm_models.items()}
    pred_df = pd.DataFrame(predictions, index=df_features.index)
    regime_vote = pred_df.mode(axis=1)[0]
    regime_vote.name = "regime"
    dist = regime_vote.value_counts(normalize=True).round(3).to_dict()
    log.info("Majority-vote regime distribution on GC: %s", dist)
    return regime_vote


# ---------------------------------------------------------------------------
# GC-specific trade simulator (15-min bars for entry + exit scanning)
# ---------------------------------------------------------------------------

def simulate_gc_trade(signal: Signal, df_15min: pd.DataFrame, qty: int = 10) -> tuple[float, str]:
    """
    Simulate a GC trade using 15-min bars (no 1-min data available).
    Entry = next bar's open. SL/TP scanned on subsequent 15-min highs/lows.
    Returns (net_pnl, outcome).
    """
    sig_time = pd.to_datetime(signal.timestamp, utc=True)
    future = df_15min[df_15min.index > sig_time]
    if len(future) < 2:
        return 0.0, "skip"

    # --- Entry: next 15-min bar open ---
    entry_bar = future.iloc[0]
    entry_price = float(entry_bar["open"])
    entry_cost = (GC_SPREAD_PER_UNIT / 2 + GC_COMMISSION + GC_SLIPPAGE) * qty

    # --- Scan subsequent bars for SL/TP ---
    scan = future.iloc[1:]
    outcome = "open"
    exit_price = float(scan["close"].iloc[-1])

    for _, bar in scan.iterrows():
        h = float(bar["high"])
        l = float(bar["low"])

        if signal.direction == "long":
            if l <= signal.stop_price:
                exit_price = signal.stop_price; outcome = "sl"; break
            elif h >= signal.tp_price:
                exit_price = signal.tp_price;  outcome = "tp"; break
        else:
            if h >= signal.stop_price:
                exit_price = signal.stop_price; outcome = "sl"; break
            elif l <= signal.tp_price:
                exit_price = signal.tp_price;  outcome = "tp"; break

    exit_cost = (GC_SPREAD_PER_UNIT / 2 + GC_COMMISSION + GC_SLIPPAGE) * qty

    if signal.direction == "long":
        gross = (exit_price - entry_price) * qty
    else:
        gross = (entry_price - exit_price) * qty

    net = gross - entry_cost - exit_cost
    return net, outcome


# ---------------------------------------------------------------------------
# GC-specific DataLoader (15-min is already the base timeframe)
# ---------------------------------------------------------------------------

class GCLoader:
    """
    Thin wrapper: loads the GC 15-min CSV directly.
    No resampling needed — data is already at 15-min resolution.
    Computes features and tags sessions for GC (UTC-based sessions).
    """
    def __init__(self):
        csv_path = CSV_PATHS.get("GC")
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(f"GC CSV not found at {csv_path}")
        self.csv_path = Path(csv_path)
        self._loader = DataLoader("GC", csv_path=self.csv_path)

    def load_features(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (df_raw_15min, df_features).
        df_raw_15min is used for exit scanning.
        df_features is used for signal generation.
        """
        # Read raw
        df = pd.read_csv(self.csv_path)
        if "symbol" in df.columns:
            df = df[df["symbol"] == "GC"].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        keep = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].astype(float)
        # Drop columns that are entirely NaN (e.g. trade_count from yfinance)
        # so that DataLoader.compute_features → dropna() doesn't erase all rows
        df = df.dropna(axis=1, how="all")

        self._loader._df_1min = df   # inject raw into loader

        log.info("GC 15-min bars loaded: %d (%s to %s)",
                 len(df), df.index[0].date(), df.index[-1].date())

        # Compute features (ATR, EMA, RSI, session, etc.)
        df_feat = self._loader.compute_features(df)
        return df, df_feat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start: Optional[str] = None, end: Optional[str] = None, qty: int = 10) -> None:
    log.info("=" * 65)
    log.info("GC TRADER — using AAPL + MSFT + SPY anchor models (majority vote)")
    log.info("=" * 65)

    # --- Load saved models ---
    hmm_models = load_anchor_models()

    # --- Load GC 15-min data ---
    gc_loader = GCLoader()
    df_raw, df_features = gc_loader.load_features()

    # --- Optional date filter ---
    if start:
        ts = pd.to_datetime(start, utc=True)
        df_features = df_features[df_features.index >= ts]
        df_raw      = df_raw[df_raw.index >= ts]
    if end:
        ts = pd.to_datetime(end, utc=True)
        df_features = df_features[df_features.index <= ts]
        df_raw      = df_raw[df_raw.index <= ts]

    log.info("Trading window: %d bars | %s to %s",
             len(df_features),
             df_features.index[0].date(),
             df_features.index[-1].date())

    # --- Regime: majority vote from 3 anchor HMMs ---
    log.info("Computing regime via majority vote...")
    regimes = majority_vote_regime(hmm_models, df_features)

    # --- Signals (no equity session filter — GC trades 24h) ---
    # Temporarily patch EQUITY_SYMBOLS + ALPACA_EXTRA_SYMBOLS to exclude GC
    # The BaselineSignalGenerator already skips the session check for non-equity symbols
    gen = BaselineSignalGenerator(symbol=GC_SYMBOL)
    signals = gen.generate(df_features, regimes)

    if not signals:
        log.warning("No signals generated on GC. Check date range or data quality.")
        return

    log.info("Signals generated: %d", len(signals))
    log.info("-" * 65)
    log.info("%-24s %-6s %-10s %-10s %-8s %-10s %-8s",
             "Timestamp", "Dir", "Rule", "Entry", "ATR", "Outcome", "Net PnL")
    log.info("-" * 65)

    results = []
    total_pnl = 0.0
    wins = losses = 0

    for sig in signals:
        net, outcome = simulate_gc_trade(sig, df_raw, qty=qty)
        total_pnl += net
        if net > 0:
            wins += 1
        elif net < 0:
            losses += 1

        results.append({
            "timestamp":   sig.timestamp,
            "direction":   sig.direction,
            "rule":        sig.rule,
            "regime":      sig.regime,
            "entry_price": sig.entry_price,
            "stop":        sig.stop_price,
            "tp":          sig.tp_price,
            "atr":         sig.atr,
            "outcome":     outcome,
            "net_pnl":     round(net, 2),
            "running_pnl": round(total_pnl, 2),
        })

        pnl_str = f"+${net:,.2f}" if net >= 0 else f"-${abs(net):,.2f}"
        log.info("%-24s %-6s %-10s $%-9.2f %-8.4f %-8s %s",
                 sig.timestamp[:19],
                 sig.direction.upper(),
                 sig.rule[:10],
                 sig.entry_price,
                 sig.atr,
                 outcome,
                 pnl_str)

    # --- Summary ---
    n = len(signals)
    win_rate = wins / n * 100 if n else 0.0
    log.info("=" * 65)
    log.info("SUMMARY — GC (Gold Futures)")
    log.info("  Data          : GC_15min_2026.csv (Jul-Sep 2026, yfinance)")
    log.info("  Model         : AAPL + MSFT + SPY HMMs (majority vote)")
    log.info("  Signals       : %d", n)
    log.info("  Wins          : %d  |  Losses: %d  |  Win Rate: %.1f%%", wins, losses, win_rate)
    log.info("  Net PnL       : $%.2f", total_pnl)
    log.info("  Qty/trade     : %d contracts", qty)
    log.info("=" * 65)
    log.info("")
    log.info("NOTE: GC used equity-trained regime models (cross-asset transfer).")
    log.info("      Previous full-history test (1h GC vs AAPL 3-fold): -$54,402")
    log.info("      Commodity assets may need a dedicated GC HMM for robustness.")

    # --- Save trade log ---
    out = Path("results") / "gc_trades.csv"
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    log.info("Trade log saved -> %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade GC using pre-trained anchor HMMs.")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end",   type=str, default=None)
    parser.add_argument("--qty",   type=int, default=10, help="Contracts per trade (default 10)")
    args = parser.parse_args()
    run(start=args.start, end=args.end, qty=args.qty)
