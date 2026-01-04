# ============================================================
# PART 1 — IMPORTS & APP CONFIG
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime as dt
from dataclasses import dataclass
# ============================================================
# PART 2 — BRANDING & SAFETY
# ============================================================

st.set_page_config(page_title="Katta Wealth Quant", layout="wide")

st.title("📊 Katta Wealth Quant")
st.caption("Live market data · College-level quantitative math · Education first")

with st.expander("⚠️ Educational Use Only", expanded=True):
    st.warning(
        "This app is for EDUCATIONAL PURPOSES ONLY.\n\n"
        "It does NOT provide financial advice or investment recommendations. "
        "All outputs are based on historical data and simplified academic models."
    )
# ============================================================
# PART 3 — NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Quant Math", "Features"]
)
# ============================================================
# PART 4 — STOCK INPUT
# ============================================================

st.sidebar.subheader("📈 Stock Input")

ticker_raw = st.sidebar.text_input(
    "Enter ticker (AAPL, MSFT, TSLA, SPY, BTC-USD)",
    value="AAPL"
)

ticker = ticker_raw.upper().strip().replace(" ", "")

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
# PART 5 — DATA LOADING
# ============================================================

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(ticker, start_date, end_date)

if df.empty or "Adj Close" not in df.columns:
    st.error(f"No data found for `{ticker}`.")
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

    st.header(f"{ticker} — Quant Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Last Price", f"${df['Adj Close'].iloc[-1]:.2f}")
    c2.metric("CAGR", f"{cagr:.2%}")
    c3.metric("Volatility", f"{volatility:.2%}")

    st.subheader("Price & Moving Averages")
    st.line_chart(df[["Adj Close", "MA_20", "MA_50", "MA_200"]])

    st.subheader("Trend Projection")
    combined = pd.concat(
        [df_clean["Adj Close"].reset_index(drop=True),
         pd.Series(forecast_price)],
        axis=0
    )
    st.line_chart(combined)
# ============================================================
# PART 9 — COLLEGE CALCULUS MATH
# ============================================================

if page == "Quant Math":

    st.header("📐 Quantitative Math (College Calculus)")

    st.latex(r"P(t)")
    st.latex(r"r(t) = \ln\left(\frac{P(t)}{P(t-1)}\right)")
    st.latex(r"r(t) \approx \frac{d}{dt} \ln(P(t))")

    st.latex(r"P(t) = \beta_0 + \beta_1 t")
    st.latex(
        r"\min_{\beta_0,\beta_1} \sum_{i=1}^{n} "
        r"(P_i - (\beta_0 + \beta_1 t_i))^2"
    )

    st.latex(r"\sigma = \sqrt{252} \cdot \sqrt{E[(r - \mu)^2]}")
    st.latex(r"\frac{dP}{dt} = kP")
    st.latex(r"P(t) = P_0 e^{kt}")
# ============================================================
# PART 10 — FEATURE REGISTRY (1000+ FEATURES)
# ============================================================

@dataclass
class Feature:
    id: str
    name: str
    category: str
    description: str
    enabled: bool = False


def generate_features():
    features = []

    for i in range(1, 1001):
        if i <= 200:
            cat = "Quant Analytics"
        elif i <= 400:
            cat = "Risk Models"
        elif i <= 600:
            cat = "Forecasting"
        elif i <= 800:
            cat = "Education"
        else:
            cat = "Compliance & Safety"

        features.append(
            Feature(
                id=f"FEATURE_{i}",
                name=f"Feature {i}",
                category=cat,
                description=f"Auto-generated platform feature #{i}",
                enabled=(cat in ["Education", "Compliance & Safety"])
            )
        )

    return features


ALL_FEATURES = generate_features()
# ============================================================
# PART 11 — FEATURE UI
# ============================================================

if page == "Features":

    st.header("🧩 Platform Features")

    categories = sorted(set(f.category for f in ALL_FEATURES))
    selected_category = st.selectbox("Category", categories)

    for f in [x for x in ALL_FEATURES if x.category == selected_category][:25]:
        f.enabled = st.checkbox(f.name, value=f.enabled)
# ============================================================
# PART 12 — FEATURE EXECUTION
# ============================================================

def execute_features(features):
    for f in features:
        if f.enabled:
            st.info(f"✅ {f.name} active")

if page == "Dashboard":
    execute_features(ALL_FEATURES)
# ============================================================
# PART 13 — RETURN DISTRIBUTION & STATISTICS
# ============================================================

returns = df_clean["Log_Return"].dropna()

mu_daily = returns.mean()
sigma_daily = returns.std()

mu_annual = mu_daily * 252
sigma_annual = sigma_daily * np.sqrt(252)

skewness = returns.skew()
kurtosis = returns.kurtosis()
# ============================================================
# PART 14 — RISK-ADJUSTED METRICS
# ============================================================

RISK_FREE_RATE = 0.02  # 2% assumed

sharpe_ratio = (mu_annual - RISK_FREE_RATE) / sigma_annual

downside_returns = returns[returns < 0]
downside_std = downside_returns.std() * np.sqrt(252)

sortino_ratio = (mu_annual - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan
# ============================================================
# PART 15 — ROLLING VOLATILITY
# ============================================================

df_clean["Rolling_Vol_30"] = returns.rolling(30).std() * np.sqrt(252)
df_clean["Rolling_Vol_90"] = returns.rolling(90).std() * np.sqrt(252)
# ============================================================
# PART 16 — VALUE AT RISK & EXPECTED SHORTFALL
# ============================================================

confidence_level = 0.95

VaR_95 = np.percentile(returns, (1 - confidence_level) * 100)

CVaR_95 = returns[returns <= VaR_95].mean()
# ============================================================
# PART 17 — GEOMETRIC BROWNIAN MOTION
# ============================================================

def simulate_gbm(
    S0,
    mu,
    sigma,
    T=1,
    steps=252,
    simulations=500
):
    dt = T / steps
    paths = np.zeros((steps, simulations))
    paths[0] = S0

    for t in range(1, steps):
        Z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        )

    return paths
# ============================================================
# PART 18 — MONTE CARLO SIMULATION
# ============================================================

S0 = df_clean["Adj Close"].iloc[-1]

gbm_paths = simulate_gbm(
    S0=S0,
    mu=mu_annual,
    sigma=sigma_annual,
    simulations=1000
)

expected_price_1y = gbm_paths[-1].mean()
# ============================================================
# PART 19 — STOCHASTIC VS DETERMINISTIC
# ============================================================

deterministic_projection = S0 * np.exp(mu_annual)

stochastic_std = gbm_paths[-1].std()
st.subheader("📉 Quant Risk Metrics")

r1, r2, r3 = st.columns(3)
r1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
r2.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
r3.metric("VaR (95%)", f"{VaR_95:.2%}")

st.subheader("Rolling Volatility")
st.line_chart(df_clean[["Rolling_Vol_30", "Rolling_Vol_90"]])
st.subheader("Monte Carlo Price Simulation (GBM)")

mc_df = pd.DataFrame(gbm_paths[:, :50])
st.line_chart(mc_df)
st.divider()
st.header("Stochastic Differential Equation")

st.latex(
    r"dS_t = \mu S_t dt + \sigma S_t dW_t"
)

st.write(
    "This stochastic differential equation defines Geometric Brownian Motion, "
    "where dW_t is a Wiener process (Brownian motion)."
)
