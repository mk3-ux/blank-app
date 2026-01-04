# ============================================================
# KATTA WEALTH QUANT
# Quantitative Finance — Educational Platform
# Live Market Data + College Calculus Math
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt

# ============================================================
# PART 1 — APP CONFIG & BRANDING
# ============================================================

APP_NAME = "Katta Wealth Quant"
APP_TAGLINE = "Quantitative insights. Education first. Risk aware."

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

st.title(APP_NAME)
st.caption(APP_TAGLINE)

# ============================================================
# PART 2 — LEGAL & SAFETY NOTICE (MANDATORY)
# ============================================================

with st.expander("⚠️ Important Notice", expanded=True):
    st.warning(
        "This application is for EDUCATIONAL PURPOSES ONLY.\n\n"
        "It does NOT provide investment advice, recommendations, or predictions. "
        "All models are simplified academic demonstrations using historical data. "
        "Markets are uncertain, and past performance does not predict future results."
    )

# ============================================================
# PART 3 — APP NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Quant Math"]
)

# ============================================================
# PART 4 — GLOBAL STOCK SELECTION (LIVE DATA)
# ============================================================

st.sidebar.subheader("📊 Stock Selection")

POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "SPY", "QQQ"
]

ticker = st.sidebar.selectbox(
    "Choose a stock",
    POPULAR_STOCKS
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=dt.date(2019, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=dt.date.today()
)

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=30,
    max_value=365,
    value=90
)

# ============================================================
# PART 5 — LIVE DATA INGESTION
# ============================================================

@st.cache_data
def load_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end, progress=False)

df = load_stock_data(ticker, start_date, end_date)

if df.empty:
    st.error("No data available for this selection.")
    st.stop()

# ============================================================
# PART 6 — QUANT METRICS
# ============================================================

df["Log_Return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
df["MA_20"] = df["Adj Close"].rolling(20).mean()
df["MA_50"] = df["Adj Close"].rolling(50).mean()
df["MA_200"] = df["Adj Close"].rolling(200).mean()

volatility = df["Log_Return"].std() * np.sqrt(252)

cagr = (
    (df["Adj Close"].iloc[-1] / df["Adj Close"].iloc[0]) **
    (252 / len(df)) - 1
)

# ============================================================
# PART 7 — LINEAR REGRESSION FORECAST
# ============================================================

df_clean = df.dropna()
t = np.arange(len(df_clean)).reshape(-1, 1)
price = df_clean["Adj Close"].values

model = LinearRegression()
model.fit(t, price)

future_t = np.arange(len(df_clean), len(df_clean) + forecast_days).reshape(-1, 1)
forecast_price = model.predict(future_t)

# ============================================================
# PART 8 — DASHBOARD PAGE
# ============================================================

if page == "Dashboard":

    st.header(f"📊 {ticker} — Live Quant Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Price", f"${df['Adj Close'].iloc[-1]:.2f}")
    col2.metric("CAGR", f"{cagr:.2%}")
    col3.metric("Volatility (σ)", f"{volatility:.2%}")

    st.subheader("Price & Moving Averages")
    st.line_chart(df[["Adj Close", "MA_20", "MA_50", "MA_200"]])

    st.subheader("Trend Projection (Linear Regression)")
    combined = pd.concat(
        [
            df_clean["Adj Close"].reset_index(drop=True),
            pd.Series(forecast_price)
        ],
        axis=0
    )
    st.line_chart(combined)

    with st.expander("View Recent Data"):
        st.dataframe(df.tail(20))

# ============================================================
# PART 9 — COLLEGE CALCULUS MATH PAGE
# ============================================================

if page == "Quant Math":

    st.header("📐 Quantitative Math (College Calculus Level)")
    st.caption(f"Mathematical framework used for {ticker}")

    st.divider()

    st.subheader("1. Price as a Continuous Function")
    st.latex(r"P(t)")
    st.write(
        "We model stock price as a continuous function of time "
        "to apply calculus-based tools."
    )

    st.divider()

    st.subheader("2. Log Returns and Derivatives")
    st.latex(r"r(t) = \ln\left(\frac{P(t)}{P(t-1)}\right)")
    st.latex(r"r(t) \approx \frac{d}{dt} \ln(P(t))")

    st.write(
        "Log returns approximate the derivative of the logarithm "
        "of price with respect to time."
    )

    st.divider()

    st.subheader("3. Linear Regression via Least Squares Optimization")
    st.latex(r"P(t) = \beta_0 + \beta_1 t")

    st.latex(
        r"\min_{\beta_0, \beta_1} "
        r"\sum_{i=1}^{n} (P_i - (\beta_0 + \beta_1 t_i))^2"
    )

    st.write(
        "The model parameters are found by minimizing squared error. "
        "This requires taking partial derivatives with respect to each parameter "
        "and solving the resulting system of equations."
    )

    st.divider()

    st.subheader("4. Volatility as Variance")
    st.latex(r"\sigma = \sqrt{252} \cdot \sqrt{E[(r - \mu)^2]}")

    st.write(
        "Volatility measures dispersion of returns and is mathematically "
        "derived from variance, which is an integral over squared deviations."
    )

    st.divider()

    st.subheader("5. Exponential Growth & Differential Equations")
    st.latex(r"\frac{dP}{dt} = kP")
    st.latex(r"P(t) = P_0 e^{kt}")

    st.write(
        "CAGR assumes proportional growth and comes from solving "
        "a first-order differential equation."
    )

    st.success(
        "This page exists to ensure mathematical transparency and "
        "to clarify that models are educational, not predictive guarantees."
    )
