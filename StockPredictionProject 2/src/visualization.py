"""
visualization.py
Chart-building functions using Plotly (interactive, Streamlit-friendly).
"""

import plotly.graph_objects as go
import pandas as pd


def plot_price_history(df: pd.DataFrame, title: str = "Stock Closing Price") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_candlestick(df: pd.DataFrame, title: str = "Candlestick Chart") -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]
    )])
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_volume(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig.update_layout(title="Trading Volume", xaxis_title="Date", yaxis_title="Volume")
    return fig


def plot_moving_averages(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    for col in ["SMA_10", "SMA_20", "SMA_50"]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(title="Price with Moving Averages", xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode="lines", name="RSI"))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(title="RSI (Relative Strength Index)", xaxis_title="Date", yaxis_title="RSI")
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram"))
    fig.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value")
    return fig


def plot_predictions_vs_actual(dates, actual, predicted, title: str = "Prediction vs Actual") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode="lines", name="Predicted"))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price")
    return fig


def plot_loss_curve(history) -> go.Figure:
    """Plots training loss for the LSTM model. `history` is a Keras History object."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history["loss"], mode="lines", name="Training Loss"))
    fig.update_layout(title="LSTM Training Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def plot_future_forecast(last_date, historical_close, future_dates, future_preds) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_close.index, y=historical_close.values,
                              mode="lines", name="Historical"))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode="lines+markers",
                              name="Forecast", line=dict(dash="dot")))
    fig.update_layout(title="Future Price Forecast", xaxis_title="Date", yaxis_title="Price")
    return fig
