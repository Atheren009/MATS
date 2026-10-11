"""Diagnostic: understand QQQ signal quality and win/loss breakdown."""
import pandas as pd
import numpy as np
from collections import Counter
from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator
from order_router.exchange import Exchange, VENUE_CONFIGS
from order_router.models import Order, OrderSide, OrderType
from order_router.price_feed import Bar
from order_router.router import OrderRouter
from order_router.routing import SmartStrategy

# --- Load ---
hmm = RegimeHMM.load(MODEL_DIR / "hmm_QQQ_full.pkl")
loader = DataLoader("QQQ")
df_1min = loader.load()
df_r = loader.resample(DEFAULT_TIMEFRAME)
df_f = loader.compute_features(df_r)

regimes = hmm.predict(df_f)
print("Regime distribution:")
print(regimes.value_counts())

gen = BaselineSignalGenerator(symbol="QQQ")
sigs = gen.generate(df_f, regimes)

directions = [s.direction for s in sigs]
rules = [s.rule for s in sigs]
reg_of_sigs = [s.regime for s in sigs]

print(f"\nTotal signals: {len(sigs)}")
print(f"Long: {directions.count('long')} | Short: {directions.count('short')}")
print(f"EMA crossover: {rules.count('ema_crossover')} | RSI reversion: {rules.count('rsi_reversion')}")
print(f"Regime of signals: {Counter(reg_of_sigs)}")

# Per-rule win rate
print("\nSample signals (first 15):")
for s in sigs[:15]:
    print(f"  {s.timestamp[:16]} {s.direction:5} {s.rule:15} regime={s.regime} "
          f"entry={s.entry_price:.2f} atr={s.atr:.4f} R={s.atr:.2f}")

# Now simulate a subset to see win rates per rule
def simulate_one(signal, df_1min, qty=500):
    sig_time = pd.to_datetime(signal.timestamp, utc=True)
    future = df_1min[df_1min.index > sig_time]
    if len(future) < 6:
        return None, "insufficient"
    entry_window = future.iloc[:5]
    venues = {name: Exchange(name, cfg, symbol=signal.symbol) for name, cfg in VENUE_CONFIGS.items()}
    side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
    order = Order(symbol=signal.symbol, side=side, order_type=OrderType.MARKET, quantity=qty)
    bars_in = [Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
                   row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
               for _, row in entry_window.iterrows()]
    res_in = OrderRouter(venues=venues, strategy=SmartStrategy()).submit(order, bars_in)
    if res_in.total_filled == 0:
        return None, "unfilled"
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
        if signal.direction == "long":
            if lows[k] <= signal.stop_price:
                raw_exit = signal.stop_price; outcome = "sl"; exit_idx = k; break
            elif highs[k] >= signal.tp_price:
                raw_exit = signal.tp_price; outcome = "tp"; exit_idx = k; break
        else:
            if highs[k] >= signal.stop_price:
                raw_exit = signal.stop_price; outcome = "sl"; exit_idx = k; break
            elif lows[k] <= signal.tp_price:
                raw_exit = signal.tp_price; outcome = "tp"; exit_idx = k; break

    exit_window = scan.iloc[exit_idx: exit_idx + 5]
    if len(exit_window) == 0:
        exit_window = scan.iloc[-1:]
    venues_out = {name: Exchange(name, cfg, symbol=signal.symbol) for name, cfg in VENUE_CONFIGS.items()}
    out_side = OrderSide.SELL if signal.direction == "long" else OrderSide.BUY
    out_order = Order(symbol=signal.symbol, side=out_side, order_type=OrderType.MARKET, quantity=filled_qty)
    bars_out = [Bar(signal.symbol, row.name, row.open, row.high, row.low, row.close,
                    row.volume, row.get("trade_count", 0), row.get("vwap", row.close))
                for _, row in exit_window.iterrows()]
    res_out = OrderRouter(venues=venues_out, strategy=SmartStrategy()).submit(out_order, bars_out)
    actual_exit = res_out.avg_fill_price if res_out.total_filled > 0 else raw_exit
    exit_cost = res_out.total_fees if res_out.total_filled > 0 else 0.0
    gross = (actual_exit - actual_entry) * filled_qty if signal.direction == "long" else (actual_entry - actual_exit) * filled_qty
    return gross - entry_cost - exit_cost, outcome

# Breakdown by direction and rule
stats = {"long_ema": [], "short_ema": [], "long_rsi": [], "short_rsi": []}
for sig in sigs:
    key = f"{sig.direction}_{sig.rule[:3]}"
    net, outcome = simulate_one(sig, df_1min)
    if net is not None:
        stats[key].append(net)

print("\n=== Per-rule breakdown ===")
for k, vals in stats.items():
    if vals:
        wins = sum(1 for v in vals if v > 0)
        total = len(vals)
        pnl = sum(vals)
        print(f"  {k:12}: {total:3} trades | WR={wins/total*100:.1f}% | PnL=${pnl:,.0f}")
