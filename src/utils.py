"""
utils.py
Shared helper functions used across modules.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta


def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_future_dates(last_date, n_days: int) -> list:
    """Generates the next n business days after last_date (skips weekends)."""
    dates = []
    current = pd.Timestamp(last_date)
    while len(dates) < n_days:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0 ... Friday=4
            dates.append(current)
    return dates


def validate_symbol(symbol: str) -> bool:
    """Basic sanity check on ticker symbol format (1-6 uppercase letters)."""
    return symbol.isalpha() and 1 <= len(symbol) <= 6
