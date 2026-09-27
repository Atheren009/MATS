"""
Central configuration for the ML trading system.

All credentials, paths, risk parameters, and cost models live here.
Phase 0: HMM + Baseline rules. LSTM/NLP deferred to Phase 1/2.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent          # b:\Order Router Python
DATA_DIR = BASE_DIR / "Price Feed"
MODEL_DIR = BASE_DIR / "ml_system" / "saved_models"
REGISTRY_DIR = BASE_DIR / "ml_system" / "registry"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Alpaca credentials (paper account)
# Override via env vars ALPACA_API_KEY / ALPACA_SECRET_KEY if preferred.
# ---------------------------------------------------------------------------

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    "PKFQLQNP2YEP3TDBLBZK4KLOQR",
)
ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY",
    "HDJDXuciMzoa9RfuLuEXRrcnS8k3pDcNmZY7qZMUzPZZ",
)
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

EQUITY_SYMBOLS = ["AAPL", "MSFT", "SPY"]
# Anchor symbols used for HMM training (the 3 original validated symbols)
ANCHOR_SYMBOLS = ["AAPL", "MSFT", "SPY"]
# QQQ is the NAS100 ETF proxy — tested 100% OOS using anchor-trained HMMs
# DIA is the Dow Jones Industrial Average ETF — traded via Alpaca 1-min data
ALPACA_EXTRA_SYMBOLS = ["QQQ", "DIA"]
GOLD_SYMBOL = "XAUUSD"
# GC is Gold Futures via yfinance 1h — tested 100% OOS
GOLD_FUTURES_SYMBOL = "GC"

ALL_SYMBOLS = EQUITY_SYMBOLS + ALPACA_EXTRA_SYMBOLS + [GOLD_SYMBOL, GOLD_FUTURES_SYMBOL]

# Local CSV paths (1-minute bars for equities; 1h for futures)
CSV_PATHS: dict[str, Path] = {
    "AAPL": DATA_DIR / "AAPL_1min_2024-2026.csv",
    "MSFT": DATA_DIR / "MSFT_1min_2024-2026.csv",
    "SPY":  DATA_DIR / "SPY_1min_2024-2026.csv",
    # QQQ — NASDAQ-100 ETF, local 1-min CSV (2024-2026, Alpaca IEX)
    "QQQ":  DATA_DIR / "QQQ_1min_2024-2026.csv",
    # DIA — Dow Jones Industrial Average ETF, fetched from Alpaca 1-min
    "DIA":  DATA_DIR / "DIA_1min_2024-2026.csv",
    # GC — Gold Futures 15-min CSV (Jul-Sep 2026, yfinance)
    "GC":   DATA_DIR / "GC_15min_2026.csv",
}

# yfinance ticker mapping (for symbols not on Alpaca)
YFINANCE_TICKERS: dict[str, str] = {
    "XAUUSD": "GC=F",   # COMEX Gold Futures — closest free proxy
    "GC":     "GC=F",   # Alias used for OOS-only gold testing
}

# ---------------------------------------------------------------------------
# Trading Parameters
# ---------------------------------------------------------------------------

PAPER_EQUITY       = 5_000.0      # Software-enforced equity cap
RISK_PER_TRADE_PCT = 0.5          # 0.5% risk per trade = $25
DEFAULT_TIMEFRAME  = "15min"
MAX_DRAWDOWN_PCT   = 10.0         # Kill switch: halt trading if equity DD > 10%

# ---------------------------------------------------------------------------
# HMM Configuration
# ---------------------------------------------------------------------------

HMM_N_STATES      = 4
HMM_COVARIANCE    = "full"
HMM_N_ITER        = 200
HMM_RANDOM_STATE  = 42
TRAIN_SPLIT_RATIO  = 0.70         # 70% train / 30% OOS

# HMM feature columns (observation vector)
HMM_FEATURES = [
    "log_return",
    "atr14_norm",
    "realised_vol_20",
    "hl_range_atr",
    "body_ratio",
    "rel_volume",
]

# ---------------------------------------------------------------------------
# Signal Parameters
# ---------------------------------------------------------------------------

EMA_FAST     = 9
EMA_SLOW     = 21
RSI_PERIOD   = 14
RSI_OVERSOLD  = 25    # Deeper extreme required before reversion long
RSI_OVERBOUGHT = 75   # Deeper extreme required before reversion short
ATR_PERIOD   = 14
ATR_STOP_MULT = 1.0   # Tighter stop: break-even win rate = 25% at 3R
ATR_TP_MULT   = 3.0   # Wider TP: 3:1 R:R, profitable above 25% win rate

# ---------------------------------------------------------------------------
# Cost Model — per instrument (DO NOT inherit NSE model)
# ---------------------------------------------------------------------------

COST_MODEL: dict[str, dict] = {
    "AAPL": {
        "spread_bps": 1.0,
        "commission_per_share": 0.0,     # Alpaca commission-free
        "slippage_bps": 1.0,
    },
    "MSFT": {
        "spread_bps": 1.0,
        "commission_per_share": 0.0,
        "slippage_bps": 1.0,
    },
    "SPY": {
        "spread_bps": 0.5,
        "commission_per_share": 0.0,
        "slippage_bps": 0.5,
    },
    # QQQ — NASDAQ-100 ETF (NAS100 proxy), commission-free on Alpaca IEX
    "QQQ": {
        "spread_bps": 0.5,
        "commission_per_share": 0.0,
        "slippage_bps": 0.5,
    },
    "XAUUSD": {
        "spread_pips": 0.35,             # Raw ECN
        "swap_long_usd_per_lot": -3.50,  # Overnight funding
        "swap_short_usd_per_lot": 1.00,
        "slippage_pips_normal": 0.5,
        "slippage_pips_news": 2.0,
    },
    # GC — Gold Futures (GC=F via yfinance 1h), OOS-only test
    "GC": {
        "spread_pips": 0.50,             # Slightly wider than XAUUSD spot
        "swap_long_usd_per_lot": -4.00,
        "swap_short_usd_per_lot": 1.50,
        "slippage_pips_normal": 0.7,
        "slippage_pips_news": 2.5,
    },
}

# ---------------------------------------------------------------------------
# LSTM Configuration (Phase 1)
# ---------------------------------------------------------------------------

LSTM_SEQ_LEN       = 64
LSTM_HIDDEN_DIM    = 32
LSTM_LAYERS        = 2
LSTM_DROPOUT       = 0.2
FORWARD_N_BARS     = 6       # Predict direction 6 bars (1.5h) into the future
LSTM_LEARNING_RATE = 1e-3
LSTM_BATCH_SIZE    = 128
LSTM_EPOCHS        = 20
LSTM_EARLY_STOP    = 5
LSTM_PROB_THRESH   = 0.5     # Decision threshold for the filter

# Note: Using generic features (PDH/PDL distance, session high/low, regime/session one-hots)
# instead of SMC features (OB/FVG) for MVP, to avoid definitional drift from live EA.

# ---------------------------------------------------------------------------
# GPU Detection (for Phase 1 LSTM)
# ---------------------------------------------------------------------------

def detect_gpu() -> dict:
    """Detect CUDA availability and return device info."""
    info = {"gpu_available": False, "device": "cpu", "gpu_name": None}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["device"] = "cuda"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 2
            )
    except ImportError:
        pass
    return info
