"""
train_combined.py
=================
Train one HMM regime model per anchor symbol (AAPL, MSFT, SPY) on the
FULL available dataset (every bar in the CSV — no holdout).

The trained models are persisted to:
    ml_system/saved_models/hmm_AAPL_full.pkl
    ml_system/saved_models/hmm_MSFT_full.pkl
    ml_system/saved_models/hmm_SPY_full.pkl

A metadata JSON is also written alongside each pkl so you can audit
exactly which bars were used and when the model was trained.

Run once (or whenever you want to refresh the models):
    python train_combined.py
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train_combined")

# Anchor symbols: trained on their own full data, used as regime classifiers
ANCHOR_SYMBOLS = ["AAPL", "MSFT", "SPY"]

# Tradeable symbols: each gets its OWN self-trained HMM (best regime accuracy)
TRADE_SYMBOLS = ["QQQ", "DIA", "DAX", "EWG"]

# All symbols to train
ALL_TRAIN_SYMBOLS = ANCHOR_SYMBOLS + TRADE_SYMBOLS


def train_and_save(symbol: str) -> Path:
    """
    Load ALL bars for *symbol*, compute features, fit HMM, save to disk.
    Returns the path to the saved .pkl file.
    """
    log.info("=" * 60)
    log.info("Training HMM on ALL data for: %s", symbol)

    loader = DataLoader(symbol)
    df_1min = loader.load()

    log.info("  Raw 1-min bars loaded: %d", len(df_1min))

    df_resampled = loader.resample(DEFAULT_TIMEFRAME)
    df_features = loader.compute_features(df_resampled)

    n_bars = len(df_features)
    date_start = df_features.index[0].strftime("%Y-%m-%d")
    date_end = df_features.index[-1].strftime("%Y-%m-%d")

    log.info("  15-min feature bars: %d  (%s → %s)", n_bars, date_start, date_end)

    # --- Fit HMM on the full feature set ---
    hmm = RegimeHMM(symbol=symbol)
    hmm.fit(df_features)

    # --- Save model ---
    model_path = MODEL_DIR / f"hmm_{symbol}_full.pkl"
    hmm.save(model_path)

    # --- Regime distribution on full data (for auditing) ---
    dist = hmm.regime_distribution(df_features)

    # --- Write metadata JSON ---
    meta = {
        "symbol": symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_bars_15min": n_bars,
        "train_bars_1min_approx": len(df_1min),
        "date_start": date_start,
        "date_end": date_end,
        "regime_distribution": dist,
        "feature_hash": hmm.feature_hash,
        "model_path": str(model_path),
    }
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("  Regime distribution: %s", dist)
    log.info("  Model saved  → %s", model_path)
    log.info("  Metadata     → %s", meta_path)

    return model_path


if __name__ == "__main__":
    saved = {}
    for sym in ALL_TRAIN_SYMBOLS:
        path = train_and_save(sym)
        saved[sym] = str(path)

    log.info("")
    log.info("=" * 60)
    log.info("ALL MODELS TRAINED AND SAVED")
    log.info("=" * 60)
    for sym, path in saved.items():
        log.info("  %-6s → %s", sym, path)
    log.info("")
    log.info("Each ticker uses its OWN self-trained HMM for maximum regime accuracy.")
    log.info("Next step: run  python trade_qqq.py  or  python trade_dia.py")
