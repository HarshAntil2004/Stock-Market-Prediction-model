"""
data_loader.py
Handles downloading historical stock data using yfinance.
"""

import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_stock_data(
    symbol: str = "AAPL",
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Downloads historical OHLCV stock data for a given symbol.

    Args:
        symbol (str): Stock ticker symbol, e.g. 'AAPL', 'TSLA'.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume],
                      indexed by Date. Empty DataFrame if download fails.
    """
    try:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if df.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return pd.DataFrame()

        # yfinance can return MultiIndex columns when multiple tickers are involved
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"

        logger.info(f"Successfully downloaded {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        return pd.DataFrame()


def save_to_csv(df: pd.DataFrame, symbol: str, folder: str = "data") -> str:
    """Saves a DataFrame to CSV inside the given folder, named after the symbol."""
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{symbol}.csv")
    df.to_csv(path)
    logger.info(f"Saved data to {path}")
    return path


if __name__ == "__main__":
    data = download_stock_data("AAPL", "2023-01-01", "2024-01-01")
    print(data.head())
    print(f"\nShape: {data.shape}")
