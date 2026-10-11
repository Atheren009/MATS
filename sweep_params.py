"""
Sweep RSI thresholds and EMA-only to find optimal signal quality for QQQ and DIA.
"""
import pandas as pd
import numpy as np
from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.models.hmm_regime import TRENDING, RANGING, NO_TRADE_REGIMES
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import SmartStrategy

def simulate_all(sigs, df_1min, qty=500):
    results = []
    for sig in sigs:
        sig_time = pd.to_datetime(sig["timestamp"], utc=True)
        future = df_1min[df_1min.index > sig_time]
        if len(future) < 6:
            continue
        entry_window = future.iloc[:5]
        venues = {name: Exchange(name, cfg, symbol=sig["symbol"]) for name, cfg in VENUE_CONFIGS.items()}
        side = OrderSide.BUY if sig["direction"] == "long" else OrderSide.SELL
        order = Order(symbol=sig["symbol"], side=side, order_type=OrderType.MARKET, quantity=qty)
        bars_in = [Bar(sig["symbol"], row.name, row.open, row.high, row.low, row.close,
                       row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
                   for _, row in entry_window.iterrows()]
        res_in = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
        if res_in.total_filled == 0:
            continue
        actual_entry = res_in.avg_fill_price
        entry_cost = res_in.total_fees
        filled_qty = res_in.total_filled

        scan = future.iloc[5:]
        highs = scan["high"].values
        lows = scan["low"].values
        exit_idx = len(scan) - 1
        raw_exit = float(scan["close"].iloc[-1])
        outcome = "open"
        for k in range(len(scan)):
            if sig["direction"] == "long":
                if lows[k] <= sig["stop"]:
                    raw_exit = sig["stop"]; outcome = "sl"; exit_idx = k; break
                elif highs[k] >= sig["tp"]:
                    raw_exit = sig["tp"]; outcome = "tp"; exit_idx = k; break
            else:
                if highs[k] >= sig["stop"]:
                    raw_exit = sig["stop"]; outcome = "sl"; exit_idx = k; break
                elif lows[k] <= sig["tp"]:
                    raw_exit = sig["tp"]; outcome = "tp"; exit_idx = k; break

        exit_window = scan.iloc[exit_idx: exit_idx + 5]
        if len(exit_window) == 0:
            exit_window = scan.iloc[-1:]
        venues_out = {name: Exchange(name, cfg, symbol=sig["symbol"]) for name, cfg in VENUE_CONFIGS.items()}
        out_side = OrderSide.SELL if sig["direction"] == "long" else OrderSide.BUY
        out_order = Order(symbol=sig["symbol"], side=out_side, order_type=OrderType.MARKET, quantity=filled_qty)
        bars_out = [Bar(sig["symbol"], row.name, row.open, row.high, row.low, row.close,
                        row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
                    for _, row in exit_window.iterrows()]
        res_out = OrderRouter(venues=venues_out, strategy=SmartStrategy()).submit(out_order, bars_out)
        actual_exit = res_out.avg_fill_price if res_out.total_filled > 0 else raw_exit
        exit_cost = res_out.total_fees if res_out.total_filled > 0 else 0.0
        gross = (actual_exit - actual_entry) * filled_qty if sig["direction"] == "long" else (actual_entry - actual_exit) * filled_qty
        results.append(gross - entry_cost - exit_cost)
    return results


def generate_signals_custom(df, regimes, symbol, atr_stop=1.0, atr_tp=3.0, 
                             rsi_os=30, rsi_ob=70, body=0.45, vol=1.1,
                             use_rsi=True, use_ema=True, trend_filter=True):
    """Generate signals with custom parameters."""
    work = df.copy()
    work["regime"] = regimes
    work["ema_fast_prev"] = work["ema_fast"].shift(1)
    work["ema_slow_prev"] = work["ema_slow"].shift(1)
    work["rsi_prev"]  = work["rsi14"].shift(1)
    work["rsi_prev2"] = work["rsi14"].shift(2)

    signals = []
    for i in range(2, len(work)):
        row = work.iloc[i]
        if row["regime"] in NO_TRADE_REGIMES:
            continue
        session = row.get("session", "regular")
        if session != "regular":
            continue
        ts = row.name
        if hasattr(ts, "tz_convert"):
            eastern = ts.tz_convert("US/Eastern")
            hhmm = eastern.hour * 100 + eastern.minute
            if hhmm < 945 or hhmm > 1515:
                continue
        atr = float(row["atr14"])
        price = float(row["close"])
        if atr < 0.001 * price:
            continue

        # EMA crossover in trending regime
        if use_ema and row["regime"] == TRENDING:
            fp, sp = row["ema_fast"], row["ema_slow"]
            fp_p, sp_p = row["ema_fast_prev"], row["ema_slow_prev"]
            if not (pd.isna(fp_p) or pd.isna(sp_p)):
                rv = row.get("rel_volume", 1.0)
                br = row.get("body_ratio", 1.0)
                tb = row.get("trend_bias", np.nan)
                if not pd.isna(rv) and rv >= vol and not pd.isna(br) and br >= body:
                    if fp_p <= sp_p and fp > sp:  # bullish crossover
                        if not trend_filter or pd.isna(tb) or tb == 1:
                            stop = price - atr * atr_stop
                            tp = price + atr * atr_tp
                            signals.append({"symbol": symbol, "timestamp": str(row.name), "direction": "long",
                                          "stop": stop, "tp": tp, "regime": row["regime"], "rule": "ema"})
                    elif fp_p >= sp_p and fp < sp:  # bearish crossover
                        if not trend_filter or pd.isna(tb) or tb == 0:
                            stop = price + atr * atr_stop
                            tp = price - atr * atr_tp
                            signals.append({"symbol": symbol, "timestamp": str(row.name), "direction": "short",
                                          "stop": stop, "tp": tp, "regime": row["regime"], "rule": "ema"})

        # RSI reversion in ranging regime
        if use_rsi and row["regime"] == RANGING:
            rsi = row["rsi14"]
            rp, rp2 = row["rsi_prev"], row["rsi_prev2"]
            if not (pd.isna(rp) or pd.isna(rp2)):
                if rp < rsi_os and rp2 < rsi_os and rsi >= rsi_os:
                    stop = price - atr * atr_stop
                    tp = price + atr * atr_tp
                    signals.append({"symbol": symbol, "timestamp": str(row.name), "direction": "long",
                                  "stop": stop, "tp": tp, "regime": row["regime"], "rule": "rsi"})
                elif rp > rsi_ob and rp2 > rsi_ob and rsi <= rsi_ob:
                    stop = price + atr * atr_stop
                    tp = price - atr * atr_tp
                    signals.append({"symbol": symbol, "timestamp": str(row.name), "direction": "short",
                                  "stop": stop, "tp": tp, "regime": row["regime"], "rule": "rsi"})
    return signals


def test_symbol(symbol, model_path, qty=500):
    hmm = RegimeHMM.load(model_path)
    loader = DataLoader(symbol)
    df_1min = loader.load()
    df_r = loader.resample(DEFAULT_TIMEFRAME)
    df_f = loader.compute_features(df_r)
    regimes = hmm.predict(df_f)

    print(f"\n{'='*60}")
    print(f"SWEEP for {symbol}")
    print(f"{'='*60}")
    configs = [
        {"label": "EMA-only + trend filter (current)", "use_rsi": False, "use_ema": True, "trend_filter": True, "rsi_os": 30, "rsi_ob": 70, "body": 0.50, "vol": 1.1},
        {"label": "EMA + RSI(25/75) + trend filter", "use_rsi": True, "use_ema": True, "trend_filter": True, "rsi_os": 25, "rsi_ob": 75, "body": 0.50, "vol": 1.1},
        {"label": "EMA + RSI(30/70) + trend filter", "use_rsi": True, "use_ema": True, "trend_filter": True, "rsi_os": 30, "rsi_ob": 70, "body": 0.50, "vol": 1.1},
        {"label": "EMA + RSI(35/65) + trend filter", "use_rsi": True, "use_ema": True, "trend_filter": True, "rsi_os": 35, "rsi_ob": 65, "body": 0.50, "vol": 1.1},
        {"label": "EMA-only NO trend filter", "use_rsi": False, "use_ema": True, "trend_filter": False, "rsi_os": 30, "rsi_ob": 70, "body": 0.50, "vol": 1.1},
        {"label": "EMA-only + trend + body=0.40", "use_rsi": False, "use_ema": True, "trend_filter": True, "rsi_os": 30, "rsi_ob": 70, "body": 0.40, "vol": 1.05},
        {"label": "EMA + RSI(35/65) + trend + body=0.40", "use_rsi": True, "use_ema": True, "trend_filter": True, "rsi_os": 35, "rsi_ob": 65, "body": 0.40, "vol": 1.05},
    ]
    for c in configs:
        sigs = generate_signals_custom(df_f, regimes, symbol, 
                                        rsi_os=c["rsi_os"], rsi_ob=c["rsi_ob"],
                                        body=c["body"], vol=c["vol"],
                                        use_rsi=c["use_rsi"], use_ema=c["use_ema"],
                                        trend_filter=c["trend_filter"])
        if not sigs:
            print(f"  {c['label']}: 0 signals")
            continue
        pnls = simulate_all(sigs, df_1min, qty=qty)
        if not pnls:
            print(f"  {c['label']}: 0 valid trades")
            continue
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = sum(abs(p) for p in pnls if p < 0)
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        wr = wins / len(pnls) * 100
        longs = sum(1 for s in sigs if s["direction"] == "long")
        shorts = sum(1 for s in sigs if s["direction"] == "short")
        print(f"  {c['label']}")
        print(f"    Trades={len(pnls)} (L={longs} S={shorts}) WR={wr:.1f}% PF={pf:.2f} PnL=${total_pnl:,.0f}")


test_symbol("QQQ", MODEL_DIR / "hmm_QQQ_full.pkl", qty=500)
test_symbol("DIA", MODEL_DIR / "hmm_DIA_full.pkl", qty=500)
