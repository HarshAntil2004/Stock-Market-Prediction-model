"""
prediction.py
Uses trained models to forecast future prices (1, 5, 30 days ahead).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def predict_next_n_days_ml(model, last_row_features: pd.DataFrame, scaler, n_days: int = 1):
    """
    Naive iterative forecast for classic ML models: repeatedly predicts the
    next value and feeds it back in as the new 'last known' Close price.
    Note: this is a simplification since we hold other features constant —
    good enough for a college project baseline, and worth discussing as a
    limitation in your report (see README).
    """
    preds = []
    current_features = last_row_features.copy()

    for _ in range(n_days):
        scaled = scaler.transform(current_features)
        pred = model.predict(scaled)[0]
        preds.append(pred)
        # Update the Open price to the new prediction for the next iteration
        current_features.iloc[0, current_features.columns.get_loc("Open")] = pred

    return preds


def predict_next_n_days_lstm(model, scaler, recent_closes: np.ndarray, window_size: int = 60, n_days: int = 1):
    """
    Predicts the next n_days closing prices using a trained LSTM model.
    recent_closes should be the most recent `window_size` closing prices.
    """
    scaled_input = scaler.transform(recent_closes.reshape(-1, 1))
    current_batch = scaled_input[-window_size:].reshape(1, window_size, 1)

    predictions = []
    for _ in range(n_days):
        next_pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(next_pred[0])
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_pred_reshaped, axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


if __name__ == "__main__":
    print("This module is meant to be imported and used with a trained model.")
    print("See app.py or main.py for a full end-to-end example.")
