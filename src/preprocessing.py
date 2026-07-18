"""
preprocessing.py
Cleans raw stock data and prepares it for feature engineering / modeling.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw OHLCV data:
    - Drops duplicate rows
    - Forward-fills missing values (appropriate for time series)
    - Drops any remaining NaN rows at the start
    """
    if df.empty:
        logger.warning("clean_data received an empty DataFrame.")
        return df

    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Forward fill is preferred over mean/median for time series:
    # it uses the last known real value instead of leaking future info.
    df = df.ffill()
    df = df.dropna()

    after = len(df)
    logger.info(f"Cleaned data: {before} -> {after} rows (duplicates/NaNs removed)")
    return df


def train_test_split_timeseries(df: pd.DataFrame, test_size: float = 0.2):
    """
    Splits time series data WITHOUT shuffling.
    Time series must preserve chronological order — the model should never
    be trained on data that comes after what it's being tested on.
    """
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test


if __name__ == "__main__":
    from data_loader import download_stock_data

    raw = download_stock_data("AAPL", "2023-01-01", "2024-01-01")
    cleaned = clean_data(raw)
    train, test = train_test_split_timeseries(cleaned)
    print(cleaned.head())
    print(f"Train: {train.shape}, Test: {test.shape}")
