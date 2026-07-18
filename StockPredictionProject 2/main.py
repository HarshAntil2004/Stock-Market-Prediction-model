"""
main.py
Command-line entry point: runs the full pipeline end-to-end
(download -> clean -> feature engineer -> train -> evaluate -> save models).

Usage:
    python main.py --symbol AAPL --start 2018-01-01 --end 2024-12-31
"""

import argparse
import logging

from src.data_loader import download_stock_data, save_to_csv
from src.preprocessing import clean_data, train_test_split_timeseries
from src.feature_engineering import generate_all_features
from src.model_training import (
    prepare_ml_data, train_and_evaluate, results_to_dataframe, save_model
)
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def run_pipeline(symbol: str, start: str, end: str):
    logger.info(f"=== Running pipeline for {symbol} ===")

    raw = download_stock_data(symbol, start, end)
    if raw.empty:
        logger.error("No data downloaded. Check the symbol and try again.")
        return

    save_to_csv(raw, symbol)

    cleaned = clean_data(raw)
    featured = generate_all_features(cleaned)
    train_df, test_df = train_test_split_timeseries(featured)

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(train_df, test_df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    comparison = results_to_dataframe(results)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))

    best_model_name = comparison.iloc[0]["Model"]
    best_model = results[best_model_name]["model"]
    save_model(best_model, f"{symbol}_{best_model_name}")
    logger.info(f"Best model: {best_model_name} (saved to models/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()

    run_pipeline(args.symbol, args.start, args.end)
