from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from datetime import datetime

API_KEY = "PKFQLQNP2YEP3TDBLBZK4KLOQR"
SECRET_KEY = "HDJDXuciMzoa9RfuLuEXRrcnS8k3pDcNmZY7qZMUzPZZ"

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