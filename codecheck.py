from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from datetime import datetime
import os

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY before running this script.")

client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Request historical bars
request_params = StockBarsRequest(
    symbol_or_symbols=["SPY"],
    timeframe=TimeFrame.Minute,
    start=datetime(2024, 1, 1),
    end=datetime(2026, 5, 21)
)

# Fetch data
bars = client.get_stock_bars(request_params)

# Convert to dataframe
df = bars.df

# Save to CSV
df.to_csv("SPY_1min_2024-2026.csv")

print("CSV file saved successfully!")
print(df.head())