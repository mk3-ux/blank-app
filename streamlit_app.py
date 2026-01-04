# ============================================================
# KATTA WEALTH QUANT
# Live Quant Dashboard — College Calculus Level
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt

# ============================================================
# PART 1 — CONFIG & BRANDING
# ============================================================

st.set_page_config(page_title="Katta Wealth Quant", layout="wide")
st.title("📊 Katta Wealth Quant")
st.caption("Live market data · College-level quantitative math · Education first")

# ============================================================
# PART 2 — SAFETY NOTICE
# ============================================================

with st.expander("⚠️ Educational Use Only", expanded=True):
    st.warning(
        "This app is for EDUCATIONAL PURPOSES ONLY.\n\n"
        "It does NOT provide financial advice or investment recommendations. "
        "All outputs are based on historical data and simplified academic models."
    )

# ============================================================
# PART 3 — NAVIGATION
# ============================================================

page = st.sidebar.radio("Navigate", ["Dashboard", "Quant Math"])

# ============================================================
# PART 4 — STOCK INPUT (TYPE ANY SYMBOL)
# ============================================================

st.sidebar.subheader("📈 Stock Input")

ticker = st.sidebar.text_input(
    "Enter stock ticker (e.g., AAPL, MSFT, TSLA, SPY)",
    value="AAPL"
).upper().strip()

start_date = st.sidebar.date_input(
    "Start Date", value=dt.date(2019, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date", value=dt.date.today()
)

forecast_days = st.sidebar.slider(
    "Forecast Days", 30, 365, 90
)

# ============================================================
# PART 5 — LOAD & CLEAN LIVE DATA
# ============================================================

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)

    # 🔴 FIX: Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df

df = load_data(ticker, start_date, end_date)

if df.empty or "Adj Close" not in df.columns:
    st.error("Invalid ticker or no data available.")
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
# PART 7 — REGRESSION FORECAST
# ============================================================

df_clean = df.dropna()
t = np.arange(len(df_clean)).reshape(-1, 1)
price = df_clean["Adj Close"].values

model = LinearRegression()
model.fit(t, price)

future_t = np.arange(len(df_clean), len(df_clean) + forecast_days).reshape(-1, 1)
forecast_price = model.predict(future_t)

# ============================================================
# PART 8 — DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(f"📊 {ticker} — Quant Dashboard")

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

    with st.expander("Recent Data"):
        st.dataframe(df.tail(20))

# ============================================================
# PART 9 — COLLEGE CALCULUS MATH PAGE
# ============================================================

if page == "Quant Math":

    st.header("📐 Quantitative Math (College Calculus)")
    st.caption(f"Mathematical framework applied to {ticker}")

    st.divider()

    st.latex(r"P(t)")
    st.write("Stock price modeled as a continuous function of time.")

    st.latex(r"r(t) = \ln\left(\frac{P(t)}{P(t-1)}\right)")
    st.latex(r"r(t) \approx \frac{d}{dt} \ln(P(t))")

    st.latex(r"P(t) = \beta_0 + \beta_1 t")
    st.latex(
        r"\min_{\beta_0,\beta_1} \sum_{i=1}^n "
        r"(P_i - (\beta_0 + \beta_1 t_i))^2"
    )

    st.latex(r"\sigma = \sqrt{252} \cdot \sqrt{E[(r - \mu)^2]}")

    st.latex(r"\frac{dP}{dt} = kP")
    st.latex(r"P(t) = P_0 e^{kt}")

    st.success(
        "All math shown is for educational transparency. "
        "Models are simplified and not predictive guarantees."
    )
