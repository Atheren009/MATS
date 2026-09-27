# XAUUSD / Forex — LSTM + HMM + NLP Self-Improving Model: Design & Critique

> **Hard rule in scope throughout:** Edge before automation. No layer gets stacked or automated without passing its own out-of-sample gate first.

---

## 1. Feasibility Check — What's Sound, What Needs Trimming

### 1.1 HMM as Regime Classifier ✅ Well-justified

Gold and forex have structurally different intraday behaviour across sessions. An HMM on price/vol/session features is appropriate here — the latent-state assumption (unobserved "market mode" driving observed returns) maps cleanly to Asian chop → London expansion → NY continuation/reversal. **No concern with this piece.**

One calibration risk: HMM with too many hidden states on a noisy price series overfits state sequences that don't generalise. **Recommended: start with 3–4 states maximum.** More states require much longer history to stabilise emission parameters.

### 1.2 LSTM per Regime — Partially Over-Engineered at MVP Stage ⚠️

Training *fully separate* LSTM weights per regime sounds principled but introduces real problems at MVP:

| Issue | Detail |
|---|---|
| **Data fragmentation** | If HMM produces 4 regimes and you train separate weights, each model trains on a fraction of already-limited labelled signal |
| **Regime boundary instability** | Live regime transitions mid-trade leave ambiguous model assignments |
| **Maintenance surface** | 4× the hyperparameter tuning, 4× the promotion gate runs |

**Better approach for MVP:** Single LSTM, regime label as an explicit one-hot/embedding input feature. Reserve separate weights for v2 after you have at least 3 years of per-regime out-of-sample data. The "test both" direction is fine for a research sprint but the architecture decision should be made *before* building the promotion gate around it.

### 1.3 NLP Sentiment Layer — Honest and Appropriately Scoped ✅

The framing (pick one use, test independently, document a null result as legitimate) is exactly right. The two paths have different complexity:

- **LSTM feature input**: Low-overhead if sentiment scoring pipeline already exists; risk is that LSTM attribution becomes muddy — did regime or sentiment drive the prediction?
- **Standalone signal**: Cleaner to validate, harder to operationalise at sub-hourly horizons

**Recommended start:** Standalone signal first. Score pre/post NFP/CPI/FOMC releases, run SPA + block bootstrap against a "no news" baseline. If it clears, *then* wire it as an LSTM input. This order preserves falsifiability.

### 1.4 Self-Improving Loop — Sound, One Gap to Address

The retrain trigger (HMM regime shift OR Page-Hinkley breach) is correct. The gap: **regime shift alone is an ambiguous trigger**. A shift from "trending" to "ranging" doesn't automatically mean the LSTM trained on trending regime is wrong — it may mean it simply shouldn't be active. The trigger logic should distinguish:

- **Regime shift → suppress LSTM, fall back to HMM+SMC baseline** (not retrain)
- **Performance breach within a stable regime → retrain**
- **Extended regime shift (>N bars) → retrain with new regime distribution**

Without this distinction, you'll retrain on regime-transition noise.

---

## 2. Feature Lists

### 2.1 HMM State Features

These are the *observation vector* per bar (recommended: 15m or 1H candle).

**Price/Volatility**
- Log return: `ln(close_t / close_{t-1})`
- ATR(14) normalised by close price
- Realised volatility: rolling 20-bar std of log returns
- High-Low range / ATR ratio (ranging vs expanding)
- Candle body-to-wick ratio (directional conviction proxy)

**Session / Time Structure**
- Session flag: one-hot {Asian, London, NY, Overlap} — do not use raw hour; use session-local time relative to open
- Time-in-session fraction (0→1 over each session)
- Kill zone flag: London open (02:00–05:00 NY), NY open (07:00–09:00 NY) — binary

**Liquidity / Market Structure**
- Distance to most recent swing high/low (in ATR units) — sweep proximity
- Relative volume: bar volume / 20-bar average volume (for instruments with volume data; use tick volume for MT4/MT5)
- Bid-ask spread proxy (if available from broker feed)

**Do NOT include as HMM features:**
- Raw price level (breaks stationarity)
- NLP sentiment (introduces latency/lookahead risk at this layer)
- Any feature requiring future bar information

**Target number of HMM states: 3–4**
Suggested labels (ex-post interpretation, not inputs): Trending-Directional, Range-Compression, High-Vol Breakout, Low-Liquidity (Asian drift).

---

### 2.2 LSTM Input Features

**Sequence length: 48–96 bars at 15m** (12–24 hours of market history). Test both; longer context helps around multi-session patterns but increases training time and gradient vanishing risk.

**Core sequence inputs (per bar)**
- Log return (same as HMM)
- ATR(14) normalised
- Realised vol (20-bar)
- Session one-hot (4 dims)
- Kill zone binary
- Relative volume

**ICT/SMC-derived structural features** ← this is the key differentiator from generic LSTM models
- Order Block proximity flag: is current bar within N pips of identified OB? (binary, calculated externally from MQL5 EA or Python replication)
- FVG flag: is price inside an unfilled Fair Value Gap? (binary)
- PDH/PDL distance: distance to Previous Day High/Low in ATR units
- NY/London session high/low at time T (rolling intraday)
- Sweep-then-reversal counter: did price sweep a swing high/low in prior K bars then close back? (binary)

**Regime input**
- HMM state: one-hot (4 dims) — the regime the bar is classified into

**NLP (if wired in v2)**
- Sentiment score: scalar, [-1, +1], clipped to zero outside ±6h of scheduled release window

**Output target**
- Do NOT predict price level. Predict one of:
  - N-bar forward log return sign (binary classification, simpler to gate)
  - N-bar forward log return (regression) + volatility (dual-head)
- Recommended N: 4–8 bars at 15m (1–2 hours) — matches typical ICT entry-to-TP timeframe

---

### 2.3 NLP Sources and Scoring

**Macro event sources (scheduled)**
| Source | Instruments | Priority |
|---|---|---|
| FOMC statements + press conference transcripts | XAUUSD, EURUSD, GBPUSD | High |
| NFP release text (BLS) | XAUUSD, EURUSD | High |
| CPI release text (BLS) | XAUUSD, EURUSD, GBPUSD | High |
| ECB rate decision + Lagarde presser | EURUSD, GBPUSD | High |
| BoE MPC minutes | GBPUSD | Medium |
| Fed Governor speeches (hawkish/dovish) | All | Medium |
| US Treasury auction results | XAUUSD | Low–Medium |

**Sources to exclude at MVP** (noise-to-signal ratio too high without domain-specific fine-tuning):
- Twitter/X financial sentiment
- Reddit (WSB-style sources)
- General news wire without event anchoring

**Scoring method options (ordered by complexity)**
1. **FinBERT** (HuggingFace `ProsusAI/finbert`) — pre-trained on financial text, no fine-tuning needed for MVP. Produces {positive, negative, neutral} + confidence. Fast baseline.
2. **LLM API call** (GPT-4o or Gemini) with a structured hawkish/dovish prompt — more accurate for central bank language but adds latency and API cost. Use for validation, not live scoring.
3. **Custom fine-tuned model** — only if FinBERT baseline shows promising signal in the SPA test. Don't fine-tune before validation.

**Temporal alignment rule:** Sentiment score is valid for ±2H around release timestamp. Outside that window, score = 0 (neutral). Never use post-release text as an input to a trade that was already open — this is a common lookahead bug.

---

## 3. Minimum Viable Version — Validation Sequence

This is the order of operations. Do not skip layers.

```
Phase 0 (Baseline — run this first, never retire it)
└── HMM regime filter
    + Raw ICT/SMC rules (LSRR / Liquidity Transition Zones)
    No LSTM, No NLP
    → Establishes: does HMM filtering improve SMC signal hit rate vs. unfiltered?
    → Metric: DSR, SPA, hit rate per regime, max drawdown per regime
    → Duration: minimum 6 months OOS before moving to Phase 1

Phase 1 (Add LSTM as confirmation filter)
└── Phase 0 stack
    + LSTM (regime as input, single model)
    → Establishes: does LSTM agreement lift hit rate without killing trade count?
    → Gate: LSTM challenger must beat Phase 0 baseline on DSR + SPA on expanding-window WF
    → Promotion: only if individual walk-forward folds clear, not just aggregate
    → Duration: 3 months OOS minimum

Phase 2 (Add NLP as standalone signal first)
└── Parallel track to Phase 1 — run independently
    + FinBERT sentiment scored against Phase 0 baseline
    → Establishes: does pre-release sentiment have directional signal?
    → Gate: same DSR + SPA + block bootstrap
    → If cleared: wire as LSTM input in Phase 3

Phase 3 (Stack validated layers only)
└── Phase 0 + Phase 1 LSTM + Phase 2 NLP (only if both cleared)
    → Re-run full validation on combined stack
    → Phase 0 baseline always kept live for ongoing comparison
```

> [!IMPORTANT]
> The retrain/promotion registry must be initialised in Phase 0 and log every run from the start. Retrofitting provenance tracking after the fact is fragile.

---

## 4. Integration with Existing MT4/MT5 EAs

### 4.1 Where ML Sits Relative to the EA Stack

```
┌─────────────────────────────────────────────────────────┐
│  MT4/MT5 EA (MQL5)                                      │
│  ┌──────────────────────┐  ┌───────────────────────┐    │
│  │  ICT/SMC Signal Logic │  │  Risk/Position Sizing  │    │
│  │  (LSRR, LTZ)         │  │  (existing EA logic)   │    │
│  └──────────┬───────────┘  └───────────────────────┘    │
│             │ Entry signal candidate                      │
│             ▼                                             │
│  ┌──────────────────────┐                                │
│  │  ML Gate (external)  │  ← Python microservice         │
│  │  HMM regime check    │    or ZeroMQ bridge            │
│  │  LSTM confirmation   │    or flat file poll           │
│  └──────────┬───────────┘                                │
│             │ Gate pass/fail                              │
│             ▼                                             │
│  ┌──────────────────────┐                                │
│  │  Order Execution     │                                │
│  └──────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

**Three integration options (pick based on latency requirements):**

| Option | Mechanism | Latency | Complexity | Recommended For |
|---|---|---|---|---|
| **A: ZeroMQ bridge** | MQL5 ↔ Python via ZMQ sockets | ~10–50ms | Medium | Live intraday, 15m signals |
| **B: Flat file / named pipe** | EA reads JSON signal file written by Python | ~100–500ms | Low | 1H+ signals, simpler setup |
| **C: REST microservice** | EA calls local HTTP endpoint on signal event | ~20–100ms | Medium-High | Most flexible for v2+ |

For XAUUSD 15m signals, **Option A (ZeroMQ)** is the standard choice — the DWX ZeroMQ MT4/MT5 connector is well-tested. Option B is fine for Phase 0/1 validation before going live.

### 4.2 What the EA Must Export to the ML Layer

To train and retrain the LSTM/HMM without manual data pulls:
- Bar OHLCV + tick volume (standard)
- **Signal timestamps**: when did the EA generate a candidate signal (before gate)?
- **Trade outcomes**: entry price, exit price, SL/TP, outcome — logged to CSV/DB per trade
- **Regime tag at entry**: which HMM state was active when the signal fired

MQL5 `OnTrade()` and `OnTradeTransaction()` events can write this to a flat file or SQLite DB that the Python layer reads.

### 4.3 Cost Model — Forex/Gold Specifics

Do not inherit the NSE 1.54 bps VWAP IS figure. Use instrument-specific estimates:

| Instrument | Typical Spread | Swap (overnight) | Slippage Model |
|---|---|---|---|
| XAUUSD | 0.30–0.50 pips (raw ECN) | Long: -\$2–5/lot/night, Short: +\$0.50–2 | 0.5–2 pips on fast news |
| EURUSD | 0.1–0.3 pips (raw ECN) | Depends on rate differential | 0.2–0.5 pips normal, 1–3 news |
| GBPUSD | 0.3–0.8 pips | Rate differential | 0.5–2 pips normal |
| USDJPY | 0.2–0.5 pips | Rate differential (carry) | 0.3–1 pips |

**Key forex cost differences from equities:**
- Swap/rollover applies to positions held past 17:00 NY — matters for overnight ICT setups
- "Commission-free" brokers embed full cost in spread; ECN/STP brokers charge commission separately — model the correct structure for your broker
- Gold has a significant funding cost on long positions (reflects physical lease rate) — ~\$3–6/lot/night at current rates

---

## 5. Retrain Registry Schema (Minimum Required Fields)

Every retrain event must log:

```json
{
  "retrain_id": "uuid",
  "timestamp_utc": "2026-09-17T15:30:00Z",
  "trigger_type": "page_hinkley_breach | regime_shift_extended | scheduled",
  "trigger_detail": {
    "metric": "rolling_sharpe_15d",
    "threshold": 0.5,
    "observed": 0.21
  },
  "training_window": {
    "start": "2025-06-01",
    "end": "2026-09-10"
  },
  "regime_distribution": {
    "trending": 0.38,
    "ranging": 0.41,
    "high_vol_breakout": 0.12,
    "low_liquidity": 0.09
  },
  "feature_set_hash": "sha256_of_feature_config",
  "walk_forward_folds": [
    {"fold": 1, "dsr": 1.42, "spa_p": 0.031, "promoted": true},
    {"fold": 2, "dsr": 1.18, "spa_p": 0.048, "promoted": true}
  ],
  "promotion_decision": "promoted | rejected",
  "rejection_reason": null,
  "challenger_vs_baseline_sharpe": {"challenger": 1.31, "baseline": 0.97},
  "deployed_model_hash": "sha256_of_model_weights"
}
```

---

## 6. Open Questions / Decisions Required

> [!IMPORTANT]
> **Data source for training**: What historical data source will feed the HMM/LSTM? Options: (a) MT4/MT5 history export, (b) Dukascopy tick data, (c) broker API. This affects tick volume quality and bar alignment — answer before starting Phase 0.

> [!IMPORTANT]
> **OB/FVG feature calculation**: The LSTM structural features (order blocks, FVGs) require a definition. Are these calculated by the existing MQL5 EA and exported, or replicated in Python? A Python replication risks definitional drift from the live EA logic — the EA export path is strongly preferred.

> [!WARNING]
> **NLP latency for live use**: FinBERT inference on a CPU takes ~200–500ms per document. If you're using this for 15m signals around news releases, this is acceptable. If you want sub-minute signals, you'll need a GPU instance or pre-scored lookup table. Clarify the target latency before Phase 2.

> [!NOTE]
> **Instruments prioritised for Phase 0**: Recommend starting with XAUUSD only. It has the most session/regime structure relevant to LSRR/LTZ, the highest volatility (more signal), and your existing discretionary edge is strongest there. Extend to majors in Phase 1+.

---

## 7. What This Is Not (Guardrails to Preserve)

- **The HMM is not an autonomous trade filter** until Phase 0 validation is complete. Until then, it runs in observation-only mode, logging regime labels alongside manual trades.
- **The LSTM is not a signal generator.** It is a confirmation layer that can suppress an SMC signal; it cannot generate entries the SMC logic did not already identify.
- **The NLP layer does not override structure.** A hawkish Fed statement does not override a clear liquidity sweep + order block setup — it can add confirmation weight or suppress a counter-trend trade, not dictate direction independently.
- **Automated promotion requires human sign-off on the first 3 retrain cycles**, even if all quantitative gates clear. The first three cycles exist to validate that the registry, feature pipeline, and walk-forward logic are behaving correctly — not just that the numbers look good.
