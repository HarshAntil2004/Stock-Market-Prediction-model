"""
model_training.py
Trains and compares multiple ML models, plus an LSTM deep learning model.
"""

import numpy as np
import pandas as pd
import logging
import joblib
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_20", "EMA_10", "EMA_20",
    "RSI", "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower", "Daily_Return", "Rolling_Std_10"
]
TARGET_COLUMN = "Close"


def prepare_ml_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Splits into X/y for classic ML models (no scaling needed for tree models,
    but we scale anyway since Linear Regression and SVR benefit from it)."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_models() -> dict:
    """Returns a dictionary of model name -> untrained model instance."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "SVR": SVR(kernel="rbf", C=100, gamma=0.1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    except ImportError:
        logger.warning("xgboost not installed, skipping XGBoost model.")
    return models


def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains every model in get_models(), evaluates on the test set,
    and returns a results dictionary with metrics + fitted model objects.
    """
    results = {}
    for name, model in get_models().items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        results[name] = {
            "model": model,
            "predictions": preds,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
        }
        logger.info(f"{name} -> RMSE: {rmse:.4f}, R2: {r2:.4f}")

    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Converts the results dict into a clean comparison table."""
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name,
            "MAE": round(r["MAE"], 4),
            "MSE": round(r["MSE"], 4),
            "RMSE": round(r["RMSE"], 4),
            "R2": round(r["R2"], 4),
            "MAPE (%)": round(r["MAPE"], 4),
        })
    return pd.DataFrame(rows).sort_values(by="RMSE")


def save_model(model, name: str, folder: str = "models") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name.replace(' ', '_')}.joblib")
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")
    return path


def load_model(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# LSTM (Deep Learning) section
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, window_size: int = 60):
    """
    Converts a 1D array of prices into overlapping sequences for LSTM training.

    Example with window_size=3 on [10,11,12,13,14]:
      X=[10,11,12] -> y=13
      X=[11,12,13] -> y=14
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(window_size: int = 60):
    """Builds a simple 2-layer LSTM regression model using Keras."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(train_close: pd.Series, window_size: int = 60, epochs: int = 20, batch_size: int = 32):
    """
    Trains an LSTM on the closing price series.
    Returns the trained model, the fitted scaler, and training history.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(train_close.values.reshape(-1, 1))

    X_train, y_train = create_sequences(scaled, window_size)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = build_lstm_model(window_size)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model, scaler, history


if __name__ == "__main__":
    from data_loader import download_stock_data
    from preprocessing import clean_data, train_test_split_timeseries
    from feature_engineering import generate_all_features

    raw = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    cleaned = clean_data(raw)
    featured = generate_all_features(cleaned)
    train_df, test_df = train_test_split_timeseries(featured)

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(train_df, test_df)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    print(results_to_dataframe(results))
