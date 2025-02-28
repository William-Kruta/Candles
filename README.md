Candles
Candles is a Python utility that downloads, caches, and manages OHLCV (Open, High, Low, Close, Volume) candle data for a given stock ticker using yfinance. It also provides functionality for data analysis, including plotting candlestick charts, calculating historical and annualized volatility, and resampling the data by week, month, or year.

Features
Data Download & Caching:
Downloads candle data from Yahoo Finance and caches it locally in CSV format. Automatically updates data if it becomes stale.

Data Analysis:

Calculate annualized volatility based on closing prices.
Compute volatility on a yearly basis.
Resample data by week, month, or year.
Visualization:
Plot candlestick charts using mplfinance and historical volatility plots with matplotlib.

Flexible Data Management:
Easily update data manually, retrieve the latest price, and plot charts directly from your code.

Dependencies
Make sure you have the following Python packages installed:

numpy
pandas
yfinance
mplfinance
matplotlib
You can install them via pip:

bash
Copy
pip install numpy pandas yfinance mplfinance matplotlib
Usage
Below is an example of how to use the Candles class:

python
Copy
import numpy as np
import pandas as pd
from candles import Candles # Assuming the file is named candles.py

# Create an instance for a ticker (e.g., Apple Inc.)

candles = Candles("AAPL", stale_threshold=3, cache_dir="M:\\CACHE\\candles", force_update=False, log=True)

# Download and set the candle data (if not cached or if data is stale)

candles.set_data(period="max", interval="1d")

# Retrieve the data as a DataFrame

df = candles.get_data()
print(df.head())

# Plot a candlestick chart

candles.plot_candles()

# Get the last closing price

last_price = candles.get_last_price()
print(f"Last Price: {last_price}")

# Calculate annualized volatility

annual_vol = candles.get_annualized_volatility()
print(f"Annualized Volatility: {annual_vol}")

# Calculate volatility by year and plot it

volatility_by_year = candles.get_volatility_by_year()
print(volatility_by_year)
candles.plot_historical_volatility()

# Resample data by month (or week, or year)

monthly_data = candles.resample_data(months=True)
print(monthly_data.head())
Class Overview
Initialization
python
Copy
candles = Candles(
ticker: str, # Stock ticker (e.g., "AAPL")
stale_threshold: int, # Days after which cached data is stale (default: 3)
cache_dir: str, # Directory for caching CSV files (default: "M:\\CACHE\\candles")
force_update: bool, # If True, forces re-download of data
log: bool # If True, prints log messages
)
Main Methods
set_data(period: str, interval: str)
Downloads or loads cached data based on the provided period and interval.

get_data(period: str, interval: str) -> pd.DataFrame
Returns the candle data, downloading it if necessary.

plot_candles()
Uses mplfinance to plot a candlestick chart of the data.

get_last_price() -> float
Returns the last closing price from the data.

get_annualized_volatility() -> float
Computes the annualized volatility from daily returns.

get_volatility_by_year() -> pd.DataFrame
Calculates and returns a DataFrame of volatility for each year.

plot_historical_volatility()
Plots the historical volatility over the years using matplotlib.

resample_data(weeks: bool, months: bool, years: bool) -> pd.DataFrame
Resamples the data based on the specified frequency (weekly, monthly, or yearly).

Contributing
Contributions and suggestions are welcome! If you have any improvements or bug fixes, please open an issue or submit a pull request.

License
MIT License

Author: William Kruta
Github: Primitive-Coding
