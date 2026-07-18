"""
feature_engineering.py
Generates technical indicators used as ML model features.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_moving_averages(df: pd.DataFrame, windows=(10, 20, 50)) -> pd.DataFrame:
    """Adds Simple Moving Average (SMA) and Exponential Moving Average (EMA) columns."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (RSI): momentum oscillator (0-100).
    Above 70 = overbought, below 30 = oversold.
    """
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence):
    difference between a fast EMA and a slow EMA, plus a signal line.
    """
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    """Bollinger Bands: SMA plus/minus N standard deviations, showing volatility."""
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = sma + (num_std * std)
    df["BB_Lower"] = sma - (num_std * std)
    df["BB_Middle"] = sma
    return df


def add_returns_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Daily returns and rolling volatility (std dev of returns)."""
    df["Daily_Return"] = df["Close"].pct_change()
    df["Rolling_Mean_10"] = df["Close"].rolling(window=10).mean()
    df["Rolling_Std_10"] = df["Close"].rolling(window=10).std()
    return df


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature engineering pipeline and drops resulting NaN rows."""
    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_returns_and_volatility(df)
    df = df.dropna()
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


if __name__ == "__main__":
    from data_loader import download_stock_data
    from preprocessing import clean_data

    raw = download_stock_data("AAPL", "2022-01-01", "2024-01-01")
    cleaned = clean_data(raw)
    featured = generate_all_features(cleaned)
    print(featured.tail())
    print(f"Columns: {list(featured.columns)}")
