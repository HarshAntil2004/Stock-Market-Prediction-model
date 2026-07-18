# AI-Based Stock Market Price Prediction System

A complete Python system that downloads historical stock data, engineers technical
indicators, trains and compares multiple Machine Learning models plus an LSTM
deep learning model, and forecasts future prices — with an interactive Streamlit
web app.

> Built college project. Screenshots below are from the
> Streamlit dashboard (`app.py`), generated on historical-style price data.

## Screenshots

**Overview tab** — closing price and trading volume

![Overview tab]<img width="1430" height="910" alt="1_overview" src="https://github.com/user-attachments/assets/01876e24-5a78-4cf7-be2f-a85731d30163" />


**Technical indicators tab** — moving averages and RSI

![Technical indicators tab]<img width="1430" height="1040" alt="2_indicators" src="https://github.com/user-attachments/assets/066c8300-8abc-4108-9ed6-dc89a078af30" />


**Model training tab** — comparison table and prediction vs actual

![Model training tab]<img width="1430" height="1040" alt="3_training" src="https://github.com/user-attachments/assets/3fe04ef8-fbc2-4d5a-a1fb-a92b00bfb1fa" />


**Forecast tab** — future price projection

![Forecast tab]<img width="1430" height="910" alt="4_forecast" src="https://github.com/user-attachments/assets/a144634c-7851-4102-b240-473154c37495" />


## Features

- Automatic historical data download via `yfinance`
- Data cleaning and time-series-safe train/test splitting
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, daily returns, rolling stats
- Model comparison: Linear Regression, Random Forest, Decision Tree, SVR, XGBoost
- Deep learning forecasting with LSTM (Keras/TensorFlow)
- Evaluation metrics: MAE, MSE, RMSE, R², MAPE
- Interactive Streamlit dashboard with charts and forecasting
- Model persistence via Joblib

## Project Structure

```
StockPredictionProject/
├── data/                   # Downloaded CSV data (generated at runtime)
├── models/                 # Saved trained models (generated at runtime)
├── report/                 # Project report / diagrams
├── src/
│   ├── data_loader.py          # Module 1: Data collection
│   ├── preprocessing.py        # Module 2: Cleaning & splitting
│   ├── feature_engineering.py  # Module 3: Technical indicators
│   ├── visualization.py        # Module 4: Charts
│   ├── model_training.py       # Module 5 & 6: ML models + LSTM
│   ├── prediction.py           # Module 8: Future forecasting
│   └── utils.py                # Shared helpers
├── app.py                  # Module 9: Streamlit web app
├── main.py                 # CLI pipeline (train/evaluate/save from terminal)
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Option A — Command line (trains and saves the best model)

```bash
python main.py --symbol AAPL --start 2018-01-01 --end 2024-12-31
```

### Option B — Streamlit web app (recommended for demos/viva)

```bash
streamlit run app.py
```

Then in the browser:
1. Pick a stock symbol and date range in the sidebar.
2. Click **Download & Train**.
3. Explore the **Overview** and **Technical Indicators** tabs.
4. In **Model Training**, compare ML models or train the LSTM.
5. In **Forecast**, view predicted prices for the next N days.

## Algorithm Overview

**Classic ML models** are trained on engineered features (moving averages, RSI,
MACD, Bollinger Bands, volume, daily returns) to predict the next closing price
as a regression problem.

**LSTM** instead learns directly from the sequence of past closing prices:

```
Window of last 60 closing prices  -->  LSTM  -->  Predicted next closing price
[p1, p2, p3, ..., p60]            -->         -->  p61
```

This sliding window is what "sequence generation" means in Module 6 — each
training example is 60 consecutive days, and the label is day 61.

## Evaluation Metrics

| Metric | Meaning |
|---|---|
| MAE  | Average absolute error, in price units |
| MSE  | Average squared error (penalizes large errors more) |
| RMSE | Square root of MSE, same units as price |
| R²   | Proportion of variance explained (closer to 1 is better) |
| MAPE | Average percentage error |

## Notes on Screenshots

The screenshots above were rendered directly from this project's own pipeline
(`preprocessing.py`, `feature_engineering.py`, `model_training.py`) using
simulated price data, then laid out to mirror the actual Streamlit tabs in
`app.py`. Run `streamlit run app.py` yourself with a real symbol (e.g. AAPL)
to see the live interactive version with real Yahoo Finance data.

## Known Limitations (worth discussing in your viva/report)

- Stock prices are highly influenced by external factors (news, macroeconomics)
  not captured by technical indicators alone.
- The iterative multi-day forecast holds some features constant, which is a
  simplification — real deployments often use walk-forward validation with
  fresh data at each step.
- Past performance of any model does not guarantee future accuracy; this
  project is for educational purposes and is not financial advice.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, yfinance, Plotly, Streamlit, Joblib, XGBoost
