"""Check actual Eastern time of QQQ signals."""
import pandas as pd
from ml_system.config import DEFAULT_TIMEFRAME, MODEL_DIR
from ml_system.data.loader import DataLoader
from ml_system.models.hmm_regime import RegimeHMM
from ml_system.signals.baseline_rules import BaselineSignalGenerator

hmm = RegimeHMM.load(MODEL_DIR / "hmm_QQQ_full.pkl")
loader = DataLoader("QQQ")
df_1min = loader.load()
df_r = loader.resample(DEFAULT_TIMEFRAME)
df_f = loader.compute_features(df_r)

regimes = hmm.predict(df_f)
gen = BaselineSignalGenerator(symbol="QQQ")
sigs = gen.generate(df_f, regimes)

print("First 20 signal timestamps (UTC vs Eastern):")
for s in sigs[:20]:
    ts = pd.to_datetime(s.timestamp, utc=True).tz_convert("US/Eastern")
    hhmm = ts.hour * 100 + ts.minute
    flag = "*** OUTSIDE WINDOW ***" if hhmm < 945 or hhmm > 1515 else ""
    print(f"  UTC={s.timestamp[:19]} ET={ts.strftime('%H:%M')} hhmm={hhmm:4d} session={s.session} dir={s.direction} {flag}")

# Count signals outside the 9:45-15:15 window
outside = []
inside = []
for s in sigs:
    ts = pd.to_datetime(s.timestamp, utc=True).tz_convert("US/Eastern")
    hhmm = ts.hour * 100 + ts.minute
    if hhmm < 945 or hhmm > 1515:
        outside.append(s)
    else:
        inside.append(s)

print(f"\nSignals OUTSIDE 9:45-15:15 ET: {len(outside)}")
print(f"Signals INSIDE  9:45-15:15 ET: {len(inside)}")
