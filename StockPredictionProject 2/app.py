"""
app.py
Streamlit web application for the AI-Based Stock Market Price Prediction System.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.data_loader import download_stock_data
from src.preprocessing import clean_data, train_test_split_timeseries
from src.feature_engineering import generate_all_features
from src.model_training import (
    prepare_ml_data, train_and_evaluate, results_to_dataframe,
    train_lstm, create_sequences
)
from src.prediction import predict_next_n_days_lstm
from src.visualization import (
    plot_price_history, plot_candlestick, plot_volume, plot_moving_averages,
    plot_rsi, plot_macd, plot_predictions_vs_actual, plot_loss_curve,
    plot_future_forecast
)
from src.utils import generate_future_dates

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 AI-Based Stock Market Price Prediction System")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Configuration")

symbol = st.sidebar.selectbox(
    "Select Stock Symbol",
    ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "NVDA"]
)

start_date = st.sidebar.date_input("Start Date", date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

model_choice = st.sidebar.selectbox(
    "Select Model Type",
    ["Compare ML Models", "LSTM (Deep Learning)"]
)

forecast_days = st.sidebar.slider("Days to Forecast", 1, 30, 5)

run_button = st.sidebar.button("Download & Train")

# ---------------- Session state ----------------
if "featured_data" not in st.session_state:
    st.session_state.featured_data = None

# ---------------- Main logic ----------------
if run_button:
    with st.spinner("Downloading and processing data..."):
        raw = download_stock_data(symbol, str(start_date), str(end_date))

    if raw.empty:
        st.error("No data found for this symbol/date range. Try a different combination.")
    else:
        cleaned = clean_data(raw)
        featured = generate_all_features(cleaned)
        st.session_state.featured_data = featured
        st.success(f"Downloaded and processed {len(featured)} rows for {symbol}.")

if st.session_state.featured_data is not None:
    df = st.session_state.featured_data

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Technical Indicators", "Model Training", "Forecast"]
    )

    with tab1:
        st.subheader(f"{symbol} Price History")
        st.plotly_chart(plot_price_history(df, f"{symbol} Closing Price"), use_container_width=True)
        st.plotly_chart(plot_candlestick(df, f"{symbol} Candlestick"), use_container_width=True)
        st.plotly_chart(plot_volume(df), use_container_width=True)
        st.dataframe(df.tail(10))

    with tab2:
        st.subheader("Technical Indicators")
        st.plotly_chart(plot_moving_averages(df), use_container_width=True)
        st.plotly_chart(plot_rsi(df), use_container_width=True)
        st.plotly_chart(plot_macd(df), use_container_width=True)

    with tab3:
        st.subheader("Model Training & Comparison")

        if model_choice == "Compare ML Models":
            if st.button("Train ML Models"):
                with st.spinner("Training models..."):
                    train_df, test_df = train_test_split_timeseries(df)
                    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(train_df, test_df)
                    results = train_and_evaluate(X_train, X_test, y_train, y_test)
                    comparison = results_to_dataframe(results)

                st.dataframe(comparison, use_container_width=True)

                best_name = comparison.iloc[0]["Model"]
                best_preds = results[best_name]["predictions"]
                st.plotly_chart(
                    plot_predictions_vs_actual(test_df.index, y_test.values, best_preds,
                                                title=f"{best_name}: Prediction vs Actual"),
                    use_container_width=True
                )

        else:  # LSTM
            epochs = st.slider("Epochs", 5, 50, 20)
            window_size = st.slider("Window Size (days)", 20, 100, 60)

            if st.button("Train LSTM Model"):
                with st.spinner("Training LSTM (this may take a minute)..."):
                    train_df, test_df = train_test_split_timeseries(df)
                    model, scaler, history = train_lstm(
                        train_df["Close"], window_size=window_size, epochs=epochs
                    )

                    st.session_state.lstm_model = model
                    st.session_state.lstm_scaler = scaler
                    st.session_state.lstm_window = window_size

                st.success("LSTM training complete.")
                st.plotly_chart(plot_loss_curve(history), use_container_width=True)

    with tab4:
        st.subheader(f"Forecast Next {forecast_days} Days")

        if "lstm_model" in st.session_state:
            model = st.session_state.lstm_model
            scaler = st.session_state.lstm_scaler
            window = st.session_state.lstm_window

            recent_closes = df["Close"].values[-window:]
            future_preds = predict_next_n_days_lstm(
                model, scaler, recent_closes, window_size=window, n_days=forecast_days
            )
            future_dates = generate_future_dates(df.index[-1], forecast_days)

            st.plotly_chart(
                plot_future_forecast(df.index[-1], df["Close"], future_dates, future_preds),
                use_container_width=True
            )

            forecast_table = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close": np.round(future_preds, 2)
            })
            st.dataframe(forecast_table, use_container_width=True)
        else:
            st.info("Train the LSTM model in the 'Model Training' tab first to see forecasts here.")

else:
    st.info("Configure your stock symbol and date range in the sidebar, then click 'Download & Train'.")
