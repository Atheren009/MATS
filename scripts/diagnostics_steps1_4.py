"""
scripts/diagnostics_steps1_4.py
================================
Steps 1-4 of the MATS edge-diagnostics plan.

Step 1 — Null-A random-direction test (500 seeds per symbol)
           Null-B (random timing) NOT run — noted in output.
Step 2 — Event study: signed forward returns at 1/3/6/12 bars, day-clustered bootstrap CI.
Step 3 — Session-spec derivation from price data (print + CSV, no code changes yet).
Step 4 — Portfolio R-multiples + exit decomposition (realised R:R vs nominal 3:1).

Notes on approximations
-----------------------
- Fast 15-min bar simulator used for Steps 1 & 4 (next-bar-open entry, scan H/L for SL/TP).
  Router fills are NOT replicated here — too slow for 500-seed sweeps.
  Cost: fixed 2 bps/side for all equity symbols; 0 for GC (pips handled separately).
  For Step 4 realized R:R this is directionally correct; exact $ figures differ from router runs.
- HMM decode is full-batch Viterbi on each symbol's full history (lookahead bias known,
  documented in Phase-B backlog).
- 500 seeds used for Null-A (matches spec; runtime ~4-8 min on CPU).
"""
from __future__ import annotations

import os, sys, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_system.config import MODEL_DIR, CSV_PATHS, DEFAULT_TIMEFRAME
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator
from ml_system.stats import deflated_sharpe_ratio

logging.basicConfig(level=logging.WARNING)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["AAPL", "MSFT", "SPY", "QQQ", "DIA", "DAX", "EWG", "GC"]
QTY     = {s: 500 for s in SYMBOLS}
QTY["GC"] = 10

ATR_STOP  = 1.0
ATR_TP    = 3.0
COST_BPS  = 2.0   # per side, equity (0 for GC — pips cost already tiny at 10 lots)
N_SEEDS   = 500
N_BOOT    = 5_000   # bootstrap iterations for CI

# ─────────────────────────────────────────────────────────────────────────────
# Signal loading
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(symbol: str):
    """Return (signals, df_15min) for a symbol using saved HMMs."""
    loader = DataLoader(symbol)

    if symbol == "GC":
        # GC data is already 15-min
        df_raw      = loader.load()
        df_features = loader.compute_features(df_raw)
        df_15min    = df_raw
        # Anchor majority-vote from AAPL/MSFT/SPY HMMs
        preds = []
        for anchor in ["AAPL", "MSFT", "SPY"]:
            hmm = RegimeHMM.load(MODEL_DIR / f"hmm_{anchor}_full.pkl")
            preds.append(hmm.predict(df_features))
        regimes = pd.concat(preds, axis=1).mode(axis=1)[0]
        regimes.name = "regime"
    else:
        df_1min     = loader.load()
        df_15min    = loader.resample(DEFAULT_TIMEFRAME)
        df_features = loader.compute_features(df_15min)
        pkl = MODEL_DIR / f"hmm_{symbol}_full.pkl"
        if not pkl.exists():
            pkl = MODEL_DIR / "hmm_SPY_full.pkl"   # fallback
        regimes = RegimeHMM.load(pkl).predict(df_features)

    # DIA was trained with relaxed thresholds (from previous run)
    if symbol == "DIA":
        gen = BaselineSignalGenerator(symbol, body_confirm=0.40, vol_confirm=1.05)
    else:
        gen = BaselineSignalGenerator(symbol)

    signals = gen.generate(df_features, regimes)
    return signals, df_15min, df_features


# ─────────────────────────────────────────────────────────────────────────────
# Fast 15-min simulator (next-bar-open entry, H/L scan for SL/TP)
# ─────────────────────────────────────────────────────────────────────────────

def fast_sim(
    ts: "pd.Timestamp",
    direction: str,
    atr: float,
    df_15: pd.DataFrame,
    qty: int,
    cost_bps: float = COST_BPS,
) -> tuple[float, str]:
    future = df_15[df_15.index > ts]
    if len(future) < 2:
        return 0.0, "skip"

    entry = float(future.iloc[0]["open"])
    stop  = (entry - atr * ATR_STOP) if direction == "long" else (entry + atr * ATR_STOP)
    tp    = (entry + atr * ATR_TP)   if direction == "long" else (entry - atr * ATR_TP)

    outcome    = "open"
    exit_price = float(future.iloc[-1]["close"])

    for _, bar in future.iloc[1:].iterrows():
        h, l = float(bar["high"]), float(bar["low"])
        if direction == "long":
            if l <= stop:   exit_price = stop; outcome = "sl"; break
            elif h >= tp:   exit_price = tp;   outcome = "tp"; break
        else:
            if h >= stop:   exit_price = stop; outcome = "sl"; break
            elif l <= tp:   exit_price = tp;   outcome = "tp"; break

    gross = ((exit_price - entry) if direction == "long" else (entry - exit_price)) * qty
    cost  = abs(entry) * qty * cost_bps / 10_000 * 2
    return round(gross - cost, 2), outcome


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Null-A random direction
# ─────────────────────────────────────────────────────────────────────────────

def step1_null_a(all_signals: dict, all_15min: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 1: Null-A Random-Direction Test  (500 seeds, seed0=42)")
    print("  Null-B (random timing) NOT implemented this run.")
    print("=" * 70)

    rows = []
    rng  = np.random.default_rng(42)

    for sym in SYMBOLS:
        signals = all_signals.get(sym, [])
        df_15   = all_15min.get(sym, pd.DataFrame())
        qty     = QTY[sym]
        cbps    = 0.0 if sym == "GC" else COST_BPS

        if not signals or df_15.empty:
            print(f"  {sym:4s}: no signals, skip")
            continue

        ts_list  = [pd.to_datetime(s.timestamp, utc=True) for s in signals]
        dir_list = [s.direction for s in signals]
        atr_list = [s.atr       for s in signals]

        # Real PnL via fast sim
        real_pnls = [fast_sim(ts, d, a, df_15, qty, cbps)[0]
                     for ts, d, a in zip(ts_list, dir_list, atr_list)]
        real_total = sum(real_pnls)
        real_wr    = sum(p > 0 for p in real_pnls) / len(real_pnls) * 100

        # Null-A: random directions, 500 seeds
        null_totals = []
        for _ in range(N_SEEDS):
            rand_dirs = rng.choice(["long", "short"], size=len(signals))
            null_totals.append(sum(
                fast_sim(ts, d, a, df_15, qty, cbps)[0]
                for ts, d, a in zip(ts_list, rand_dirs, atr_list)
            ))

        null_arr = np.array(null_totals)
        p_val    = float(np.mean(null_arr >= real_total))
        pct      = float(np.mean(null_arr < real_total) * 100)

        print(f"  {sym:4s}: n={len(signals):3d}  real=${real_total:>9,.0f}  "
              f"null=[{np.percentile(null_arr,5):>9,.0f}, {np.percentile(null_arr,95):>9,.0f}]  "
              f"pct={pct:4.1f}  p={p_val:.3f}")

        rows.append({
            "Symbol":     sym,
            "N_Signals":  len(signals),
            "Real_PnL":   round(real_total),
            "Real_WR%":   round(real_wr, 1),
            "Null_P5":    round(float(np.percentile(null_arr, 5))),
            "Null_Med":   round(float(np.median(null_arr))),
            "Null_P95":   round(float(np.percentile(null_arr, 95))),
            "Percentile": round(pct, 1),
            "p_value":    round(p_val, 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "null_test.csv", index=False)
    print(f"\n  -> results/null_test.csv")

    # Pooled
    pooled_sigs = sum(r["N_Signals"] for r in rows)
    pooled_real = sum(r["Real_PnL"] for r in rows)
    print(f"\n  POOLED:  {pooled_sigs} signals,  Real_PnL=${pooled_real:,.0f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Event study: forward returns
# ─────────────────────────────────────────────────────────────────────────────

def step2_event_study(all_signals: dict, all_15min: dict) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 2: Event Study — Forward Returns (next-bar-open entry)")
    print("  Horizons: 1, 3, 6, 12 bars (15-min).  Net = gross - 2*2bps.")
    print("  CI method: day-clustered block bootstrap (5000 iterations, seed=42).")
    print("=" * 70)

    HORIZONS = [1, 3, 6, 12]
    records  = []

    for sym in SYMBOLS:
        for sig in all_signals.get(sym, []):
            ts     = pd.to_datetime(sig.timestamp, utc=True)
            df_15  = all_15min[sym]
            future = df_15[df_15.index > ts]
            if len(future) < 2:
                continue
            entry = float(future.iloc[0]["open"])
            row   = {
                "symbol": sym, "date": ts.date(),
                "direction": sig.direction, "rule": sig.rule, "regime": sig.regime,
            }
            for h in HORIZONS:
                if len(future) > h:
                    ep  = float(future.iloc[h]["close"])
                    raw = (ep - entry) / entry
                    sgn = raw if sig.direction == "long" else -raw
                    row[f"g{h}"] = round(sgn * 100, 5)
                    row[f"n{h}"] = round((sgn - COST_BPS * 2 / 10_000) * 100, 5)
                else:
                    row[f"g{h}"] = row[f"n{h}"] = np.nan
            records.append(row)

    if not records:
        print("  No records.")
        return pd.DataFrame()

    ev = pd.DataFrame(records)
    ev.to_csv(RESULTS_DIR / "event_study_raw.csv", index=False)

    def cb_ci(series: pd.Series, dates: pd.Series, n=N_BOOT):
        """Day-clustered block bootstrap 95% CI."""
        valid  = series.dropna()
        d_vals = dates.loc[valid.index]
        udates = d_vals.unique()
        if len(udates) < 5:
            return float(valid.mean()), np.nan, np.nan
        rng  = np.random.default_rng(42)
        boot = [
            pd.concat([valid[d_vals == d]
                       for d in rng.choice(udates, size=len(udates), replace=True)]).mean()
            for _ in range(n)
        ]
        return float(valid.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    summary = []
    # Pooled
    for h in HORIZONS:
        col = f"n{h}"
        m, lo, hi = cb_ci(ev[col], ev["date"])
        summary.append({"Cut": "POOLED", "Sub": "ALL",
                        "H": f"{h}bar", "N": int(ev[col].notna().sum()),
                        "Mean%": round(m, 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})
    # By rule
    for rule in ev["rule"].unique():
        sub = ev[ev["rule"] == rule]
        for h in HORIZONS:
            col = f"n{h}"
            if sub[col].notna().sum() < 5:
                continue
            m, lo, hi = cb_ci(sub[col], sub["date"])
            summary.append({"Cut": "rule", "Sub": rule,
                            "H": f"{h}bar", "N": int(sub[col].notna().sum()),
                            "Mean%": round(m, 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})
    # By symbol (1-bar only for brevity)
    for sym in ev["symbol"].unique():
        sub = ev[ev["symbol"] == sym]
        col = "n1"
        if sub[col].notna().sum() < 5:
            continue
        m, lo, hi = cb_ci(sub[col], sub["date"])
        summary.append({"Cut": "symbol", "Sub": sym,
                        "H": "1bar", "N": int(sub[col].notna().sum()),
                        "Mean%": round(m, 4), "CI_lo": round(lo, 4), "CI_hi": round(hi, 4)})

    df_s = pd.DataFrame(summary)
    df_s.to_csv(RESULTS_DIR / "event_study_summary.csv", index=False)

    print(f"\n  {'Cut':<8} {'Sub':<14} {'H':<6} {'N':>5}  {'Mean%':>8}  {'95% CI':>20}")
    for _, r in df_s.iterrows():
        ci = f"[{r['CI_lo']:>8.4f}, {r['CI_hi']:>8.4f}]"
        print(f"  {r['Cut']:<8} {r['Sub']:<14} {r['H']:<6} {r['N']:>5}  {r['Mean%']:>8.4f}  {ci}")

    print(f"\n  -> results/event_study_raw.csv   ({len(ev)} rows)")
    print(f"  -> results/event_study_summary.csv")
    return df_s


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Session spec derivation from raw 1-min data
# ─────────────────────────────────────────────────────────────────────────────

def step3_session_spec() -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("STEP 3: Session Spec — Derived from Raw CSV Hour Distribution")
    print("  This step PRINTS only. No code changes yet.")
    print("=" * 70)

    rows = []
    for sym in SYMBOLS:
        csv_path = CSV_PATHS.get(sym)
        if not csv_path or not Path(csv_path).exists():
            print(f"  {sym:4s}: CSV not found, skip")
            continue

        df_raw = pd.read_csv(csv_path, parse_dates=["timestamp"])
        if df_raw["timestamp"].dt.tz is None:
            df_raw["timestamp"] = df_raw["timestamp"].dt.tz_localize("UTC")
        else:
            df_raw["timestamp"] = df_raw["timestamp"].dt.tz_convert("UTC")

        df_raw = df_raw.sort_values("timestamp")
        n_bars = len(df_raw)

        if sym == "GC":
            # GC bars are 15-min, already UTC
            hours = df_raw["timestamp"].dt.hour
            tz_label = "UTC"
        else:
            et    = df_raw["timestamp"].dt.tz_convert("US/Eastern")
            hours = et.dt.hour
            tz_label = "US/Eastern"

        hcounts   = hours.value_counts().sort_index()
        threshold = n_bars * 0.005   # hours with >0.5% of all bars
        active    = sorted(hcounts[hcounts >= threshold].index.tolist())

        h_open  = min(active) if active else "?"
        h_close = max(active) if active else "?"

        # What the code currently does
        if sym in ["DAX", "EWG"]:
            current = "9:45-15:15 ET (US equity filter — applied to EU ETF, likely suppressing bars)"
            needs_fix = True
        elif sym == "GC":
            current = "No session filter (trade_gc loads GC_15min_2026.csv directly)"
            needs_fix = False   # already unrestricted; maintenance window is the only concern
        else:
            current = "9:45-15:15 ET"
            needs_fix = False

        # Correct window from data
        if sym == "GC":
            correct = f"CME GC: 17:00-16:00 UTC Sun-Fri; excl 16:00-17:00 maintenance"
        else:
            correct = f"{tz_label} hours {h_open}:00-{h_close}:59 (data-derived; entry window +15 / -15 min)"

        print(f"  {sym:4s}: bars={n_bars:>7,}  active_hours({tz_label})={active}  fix={needs_fix}")
        print(f"         current='{current}'")
        print(f"         correct='{correct}'")

        rows.append({
            "Symbol":       sym,
            "Total_Bars":   n_bars,
            "TZ":           tz_label,
            "Active_Hours": str(active),
            "H_Open":       h_open,
            "H_Close":      h_close,
            "Current_Code": current,
            "Derived_Spec": correct,
            "Needs_Fix":    needs_fix,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "session_spec.csv", index=False)
    print(f"\n  -> results/session_spec.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Portfolio R-multiples + exit decomposition
# ─────────────────────────────────────────────────────────────────────────────

def step4_portfolio(all_signals: dict, all_15min: dict) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 70)
    print("STEP 4: Portfolio R-multiples + Exit Decomposition")
    print("  Same fast simulator as Step 1.  n_trials=8 for DSR (one per symbol).")
    print("=" * 70)

    trade_rows = []
    for sym in SYMBOLS:
        qty  = QTY[sym]
        cbps = 0.0 if sym == "GC" else COST_BPS
        for sig in all_signals.get(sym, []):
            ts     = pd.to_datetime(sig.timestamp, utc=True)
            df_15  = all_15min.get(sym, pd.DataFrame())
            if df_15.empty:
                continue
            future = df_15[df_15.index > ts]
            if len(future) < 2:
                continue
            entry = float(future.iloc[0]["open"])
            pnl, outcome = fast_sim(ts, sig.direction, sig.atr, df_15, qty, cbps)
            init_risk = sig.atr * ATR_STOP * qty
            r_mult    = pnl / init_risk if init_risk > 0 else np.nan
            trade_rows.append({
                "symbol":    sym,
                "date":      ts.date(),
                "direction": sig.direction,
                "outcome":   outcome,
                "pnl":       pnl,
                "init_risk": init_risk,
                "r_mult":    r_mult,
            })

    if not trade_rows:
        print("  No trades found.")
        return pd.DataFrame(), {}

    df = pd.DataFrame(trade_rows)
    df.to_csv(RESULTS_DIR / "portfolio_r_multiples.csv", index=False)

    # ---- Portfolio stats ----
    daily  = df.groupby("date")["pnl"].sum().sort_index()
    std    = daily.std()
    sharpe = (daily.mean() / std * np.sqrt(252)) if std > 0 else np.nan

    cum     = daily.cumsum()
    max_dd  = float((cum - cum.expanding().max()).min())
    net_pnl = df["pnl"].sum()
    n_tot   = len(df)
    n_wins  = int((df["pnl"] > 0).sum())
    wr      = n_wins / n_tot * 100 if n_tot else 0
    mean_r  = float(df["r_mult"].mean())

    # DSR — daily returns, n_trials=8
    dsr = deflated_sharpe_ratio(daily, n_trials=8, periods_per_year=252)

    print(f"\n  Pooled portfolio ({n_tot} trades across 8 symbols):")
    print(f"  Win Rate={wr:.1f}%  Mean R={mean_r:.3f}  Ann.Sharpe={sharpe:.2f}  MaxDD=${max_dd:,.0f}  DSR={dsr:.3f}")
    print(f"  Net PnL (fast sim) = ${net_pnl:,.0f}")

    # ---- Exit decomposition ----
    ex_rows = []
    for oc in df["outcome"].unique():
        sub  = df[df["outcome"] == oc]
        n    = len(sub)
        wins = int((sub["pnl"] > 0).sum())
        ex_rows.append({
            "Outcome":   oc,
            "Count":     n,
            "Share%":    round(n / n_tot * 100, 1),
            "Win%":      round(wins / n * 100, 1) if n else 0,
            "Mean_R":    round(sub["r_mult"].mean(), 3),
            "Mean_PnL":  round(sub["pnl"].mean()),
        })
    df_ex = pd.DataFrame(ex_rows).sort_values("Count", ascending=False).reset_index(drop=True)
    df_ex.to_csv(RESULTS_DIR / "exit_decomposition.csv", index=False)

    # Realized R:R
    tp_mean = df[df["outcome"] == "tp"]["r_mult"].mean() if (df["outcome"] == "tp").any() else 0
    sl_mean = df[df["outcome"] == "sl"]["r_mult"].mean() if (df["outcome"] == "sl").any() else 0
    open_mean = df[df["outcome"] == "open"]["r_mult"].mean() if (df["outcome"] == "open").any() else 0
    realized_rr = abs(tp_mean / sl_mean) if sl_mean else np.nan

    print(f"\n  Exit Decomposition:")
    print(f"  {'Outcome':<8} {'Count':>6} {'Share%':>7} {'Win%':>7} {'Mean_R':>8} {'Mean_PnL':>10}")
    for _, r in df_ex.iterrows():
        print(f"  {r['Outcome']:<8} {r['Count']:>6} {r['Share%']:>7.1f} {r['Win%']:>7.1f} "
              f"{r['Mean_R']:>8.3f} {r['Mean_PnL']:>10,.0f}")

    print(f"\n  Nominal R:R = {ATR_TP/ATR_STOP:.1f}:1")
    print(f"  Realized R:R (avg TP-exit R / avg SL-exit R) = {realized_rr:.2f}:1" if not np.isnan(realized_rr) else "  Realized R:R = N/A")

    summary = {
        "Trades": n_tot, "Win%": round(wr, 1), "Mean_R": round(mean_r, 3),
        "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else "N/A",
        "MaxDD_$": round(max_dd), "Net_PnL": round(net_pnl),
        "DSR_8trials": round(dsr, 3),
        "Realized_RR": round(realized_rr, 2) if not np.isnan(realized_rr) else "N/A",
        "Nominal_RR":  f"{ATR_TP/ATR_STOP:.1f}:1",
        "TP_mean_R": round(tp_mean, 3), "SL_mean_R": round(sl_mean, 3),
        "OPEN_mean_R": round(open_mean, 3),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "portfolio_summary.csv", index=False)
    print(f"\n  -> results/portfolio_r_multiples.csv  ({len(df)} rows)")
    print(f"  -> results/exit_decomposition.csv")
    print(f"  -> results/portfolio_summary.csv")
    return df_ex, summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading signals for 8 symbols via saved HMMs ...")
    all_signals: dict = {}
    all_15min:   dict = {}

    for sym in SYMBOLS:
        print(f"  {sym} ...", end=" ", flush=True)
        try:
            sigs, df15, _ = load_signals(sym)
            all_signals[sym] = sigs
            all_15min[sym]   = df15
            print(f"ok ({len(sigs)} signals)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            all_signals[sym] = []
            all_15min[sym]   = pd.DataFrame()

    step1_null_a(all_signals, all_15min)
    step2_event_study(all_signals, all_15min)
    step3_session_spec()
    step4_portfolio(all_signals, all_15min)

    print("\n\nAll output files:")
    for f in [
        "null_test.csv", "event_study_summary.csv", "event_study_raw.csv",
        "session_spec.csv", "portfolio_r_multiples.csv",
        "exit_decomposition.csv", "portfolio_summary.csv",
    ]:
        p = RESULTS_DIR / f
        size = p.stat().st_size if p.exists() else -1
        print(f"  {f:<35} {size:>8} bytes" if size >= 0 else f"  {f:<35} MISSING")
