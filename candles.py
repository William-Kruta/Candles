import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from pathlib import Path


class Candles:
    """
    A class to download, cache, and manage OHLCV candle data for a given ticker using yfinance.

    Attributes:
        ticker (str): The stock ticker symbol.
        stale_threshold (int): Number of days after which cached data is considered stale.
        cache_dir (str): Directory path for caching candle data.
        force_update (bool): If True, forces re-download of data.
        log (bool): If True, prints log messages.
        data (pd.DataFrame): The DataFrame containing candle data.
        date_format (str): Format used for dates.
    """

    def __init__(
        self,
        ticker: str,
        stale_threshold: int = 3,
        cache_dir: str = "M:\\CACHE\\candles",
        force_update: bool = False,
        log: bool = True,
    ):
        self.ticker = ticker.upper()
        self.stale_threshold = stale_threshold
        self.cache_dir = cache_dir
        self.force_update = force_update
        self.log = log
        self.data = pd.DataFrame()
        self.date_format = "%Y-%m-%d"

    def set_data(
        self,
        period: str = "max",
        interval: str = "1d",
    ):
        """
        Download candle data or load from cache if available and not stale.

        Parameters:
            period (str): Data period to download.
            interval (str): Data interval.
        """
        path = Path(self.cache_dir) / f"{self.ticker}.csv"
        try:
            if self.force_update:
                df = self._download_candles(period, interval)
                df.to_csv(path)
                if self.log:
                    print(f"Updated {self.ticker} candles.")
            else:
                # Read from cache
                df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
                last_date = df.index[-1]
                if self.check_staleness(last_date):
                    df = self._download_candles(period, interval)
                    df.to_csv(path)
                    if self.log:
                        print(f"Downloaded updated {self.ticker} candles.")
                else:
                    if self.log:
                        print(f"Using cached {self.ticker} data.")
        except FileNotFoundError:
            # If the file doesn't exist, download data and create cache directory if needed.
            df = self._download_candles(period, interval)
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
            if self.log:
                print(f"Downloaded {self.ticker} candles.")
        self.data = df

    def hard_set_data(self, new_data: pd.DataFrame):
        """
        Directly set the candle data.

        Parameters:
            new_data (pd.DataFrame): New candle data.
        """
        self.data = new_data

    def get_data(self, period: str = "max", interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve the candle data, downloading it if necessary.

        Parameters:
            period (str): Data period.
            interval (str): Data interval.

        Returns:
            pd.DataFrame: The candle data.
        """
        if self.data.empty:
            self.set_data(period, interval)
        return self.data

    def plot_candles(self):
        import mplfinance as mpf

        df = self.get_data()
        mpf.plot(df, type="candle", style="charles", title=f"{self.ticker} Candles")

    def get_last_price(self) -> float:
        """
        Get the last closing price from the candle data.

        Returns:
            float: Last close price.
        """
        df = self.get_data()
        return df["Close"].iloc[-1]

    def _download_candles(self, period: str, interval: str) -> pd.DataFrame:
        """
        Download candle data from yfinance.

        Parameters:
            period (str): Data period.
            interval (str): Data interval.

        Returns:
            pd.DataFrame: Downloaded candle data.
        """
        candles = yf.download(
            self.ticker, period=period, interval=interval, progress=False
        )
        return candles

    def check_staleness(self, last_recorded_date) -> bool:
        """
        Check if the cached data is stale based on the last recorded date.

        Parameters:
            last_recorded_date (str or datetime): The date of the last record.

        Returns:
            bool: True if data is stale, False otherwise.
        """
        if isinstance(last_recorded_date, str):
            last_recorded_date = dt.datetime.strptime(
                last_recorded_date, self.date_format
            )
        today = dt.datetime.now().date()
        delta = today - last_recorded_date.date()
        return delta.days >= self.stale_threshold

    def get_annualized_volatility(self) -> float:
        """
        Calculate the annualized volatility from the closing prices.

        Returns:
            float: Annualized volatility.
        """
        if self.data.empty:
            self.set_data()
        daily_returns = self.data["Close"].pct_change().dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        return annualized_volatility

    def get_volatility_by_year(self):
        """
        Calculate the volatility by year from the closing prices.

        Returns:
           float: Volatility by year.
        """
        if self.data.empty:
            self.set_data()
        unique_years = self.data.index.year.unique()
        volatility_by_year = {"Year": [], "Volatility": []}
        for y in unique_years:
            data_slice = self.data[self.data.index.year == y].copy()
            data_slice["log_return"] = np.log(
                data_slice["Close"] / data_slice["Close"].shift(1)
            )
            volatility = data_slice["log_return"].std() * np.sqrt(len(data_slice))
            volatility_by_year["Year"].append(y)
            volatility_by_year["Volatility"].append(volatility)

        return pd.DataFrame(volatility_by_year).set_index("Year")

    def plot_historical_volatility(self):
        """
        Plot the historical volatility from the closing prices.
        """
        import matplotlib.pyplot as plt

        df = self.get_volatility_by_year()
        plt.plot(df.index, df["Volatility"], label="Historical Volatility")
        plt.title("Historical Volatility")
        plt.xlabel("Year")
        plt.ylabel("Volatility")
        plt.legend()
        plt.show()

    def resample_data(
        self,
        weeks: bool = False,
        months: bool = False,
        years: bool = False,
    ) -> pd.DataFrame:
        """
        Resample the candle data by week, month, or year.

        Parameters:
            weeks (bool): Resample by week if True.
            months (bool): Resample by month if True.
            years (bool): Resample by year if True.

        Returns:
            pd.DataFrame: Resampled OHLCV data.
        """
        if self.data.empty:
            self.set_data()
        data = self.data.copy()
        agg_dict = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        if weeks:
            data = data.resample("W").agg(agg_dict)
        elif months:
            data = data.resample("M").agg(agg_dict)
        elif years:
            data = data.resample("Y").agg(agg_dict)
        return data


if __name__ == "__main__":
    candles = Candles("AAPL")
