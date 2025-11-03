import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, ConstantKernel as C
from arch import arch_model
from sklearn.linear_model import LinearRegression
import ccxt
import time

st.set_page_config(layout="wide")
st.title("GPR & GARCH Volatility Dashboard")

st.sidebar.header("Settings")

symbol_names = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA"
}
symbol_name = st.sidebar.selectbox("Coin", list(symbol_names.keys()), index=1)
symbol = symbol_names[symbol_name]

currency = st.sidebar.selectbox("Quote Currency", ["USD", "EUR", "BTC"], index=0)

st.sidebar.subheader("Timeframe Selection")

time_unit = st.sidebar.selectbox("Unit", ["Minutes", "Hours", "Days"], index=1)
amount = st.sidebar.number_input(f"Number of {time_unit.lower()}", min_value=1, max_value=1000, value=200)

timeframe_map = {
    "Minutes": "1m",
    "Hours": "1h",
    "Days": "1d"
}
timeframe = timeframe_map[time_unit]

if time_unit == "Minutes":
    total_days = amount / (60 * 24)
elif time_unit == "Hours":
    total_days = amount / 24
else:
    total_days = amount

@st.cache_data(show_spinner=False)
def fetch_data_kraken(symbol, currency, days, timeframe='1h'):
    """
    Fetch OHLCV data from Kraken using ccxt.
    """
    exchange = ccxt.kraken()
    market_symbol = f"{symbol}/{currency}"
    since = exchange.parse8601((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).isoformat())
    all_data = []
    limit = 720

    while True:
        try:
            data = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, since=since, limit=limit)
            if not data:
                break
            all_data += data
            since = data[-1][0] + 1
            time.sleep(1.5)
        except Exception as e:
            st.warning(f"Error fetching data: {e}")
            break

        if len(data) < limit:
            break

    if not all_data:
        st.error(f"No data returned for {market_symbol}")
        st.stop()

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df

df = fetch_data_kraken(symbol, currency, total_days, timeframe)

def plot_gpr(df):
    data = df['close'].values
    X = np.arange(len(data)).reshape(-1, 1)
    y = data

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (
        C(1.0, (1e-3, 1e3)) *
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-7, n_restarts_optimizer=10, normalize_y=True)
    gpr.fit(X_scaled, y_scaled)

    X_pred = np.linspace(0, len(data) + 20, 300).reshape(-1, 1)
    X_pred_scaled = scaler_X.transform(X_pred)
    y_pred_scaled, sigma_scaled = gpr.predict(X_pred_scaled, return_std=True)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    price_range = scaler_y.data_max_ - scaler_y.data_min_
    sigma = sigma_scaled * price_range

    current_price = data[-1]
    predicted_price = y_pred[-1]
    st.write(f"**Current price:** {current_price:.4f} {currency.upper()}")
    st.write(f"**Predicted price at the end of forecast:** {predicted_price:.4f} {currency.upper()}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(X_pred, y_pred, "b", label="GPR Prediction")
    ax.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2, color="blue", label="95% CI")
    ax.scatter(X, y, color="black", s=20, label="Observed Prices")
    ax.set_title(f"Gaussian Process Regression on {symbol.upper()} ({currency.upper()})")
    ax.set_xlabel("Time Index")
    ax.set_ylabel(f"Price ({currency.upper()})")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    return fig

def plot_garch(df):
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    am = arch_model(df['log_return'] * 100, vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp="off")

    current_vol = res.conditional_volatility[-1]
    current_var = current_vol ** 2
    st.write(f"**Current daily volatility (%):** {current_vol:.4f}")
    st.write(f"**Current daily variance:** {current_var:.4f}")

    forecast_horizon = 10
    forecast = res.forecast(horizon=forecast_horizon)
    daily_variances = forecast.variance.values[-1]
    daily_vols = np.sqrt(daily_variances)
    annualized_daily_vols = daily_vols * np.sqrt(252)

    vol_10day = np.sqrt(daily_variances.sum())
    annualized_10day_vol = vol_10day * np.sqrt(252 / forecast_horizon)

    st.write("**Forecasted daily volatility (%):**")
    for i, v in enumerate(daily_vols, 1):
        st.write(f"h.{i}: {v:.4f}  Annualized: {annualized_daily_vols[i-1]:.4f}")

    st.write(f"**10-day volatility (%):** {vol_10day:.4f}")
    st.write(f"**10-day annualized volatility:** {annualized_10day_vol:.4f}")

    vol_series = res.conditional_volatility * np.sqrt(252)
    residuals = res.resid / res.conditional_volatility
    vol_values = vol_series.values.reshape(-1, 1)
    time_index = np.arange(len(vol_series)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(time_index, vol_values)
    trend = model.predict(time_index)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    current_trend = trend[-1][0]
    threshold = 0.05 * current_trend

    if slope > 0 and current_vol > current_trend + threshold:
        regime, color = "Rising High Volatility", "red"
    elif slope > 0 and current_vol < current_trend - threshold:
        regime, color = "Rising Low Volatility", "green"
    elif slope < 0 and current_vol > current_trend + threshold:
        regime, color = "Falling High Volatility", "orange"
    elif slope < 0 and current_vol < current_trend - threshold:
        regime, color = "Falling Low Volatility", "blue"
    else:
        regime, color = "Stable Volatility", "lightblue"

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plt.suptitle(f"GARCH(1,1) Volatility Analysis for {symbol.upper()} ({currency.upper()})", fontsize=14)

    axes[0].plot(df.index, residuals, color='steelblue', alpha=0.7)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_title("Standardized Residuals")
    axes[0].set_ylabel("Residuals")

    vol_mean = vol_series.mean()
    vol_p25 = np.percentile(vol_series, 25)
    vol_p75 = np.percentile(vol_series, 75)

    axes[1].plot(vol_series.index, vol_series, label='Annualized Volatility', color='steelblue')
    axes[1].axhline(vol_mean, linestyle='--', color='black', label=f"Mean: {vol_mean:.4f}")
    axes[1].axhline(vol_p25, linestyle='--', color='green', label=f"25th %ile: {vol_p25:.4f}")
    axes[1].axhline(vol_p75, linestyle='--', color='red', label=f"75th %ile: {vol_p75:.4f}")
    axes[1].set_title("Volatility Levels and Percentiles")
    axes[1].set_ylabel("Volatility (%)")
    axes[1].legend()

    rolling_vol = vol_series.rolling(window=5, min_periods=1).mean()

    axes[2].plot(vol_series.index, vol_series, label='Conditional Volatility', color='steelblue', alpha=0.6)
    axes[2].plot(vol_series.index, trend.flatten(), color='red', linestyle='--',
                 label=f'Trend: y={slope:.6f}x + {intercept:.6f}')

    axes[2].scatter(vol_series.index[-1], vol_series.iloc[-1], color=color, s=100, zorder=5,
                    edgecolor='black', linewidth=0.8,
                    label=f'Current Volatility ({regime})')
    axes[2].axhline(vol_mean, color='gray', linestyle='--', linewidth=1,
                    label=f"Mean Volatility: {vol_mean:.2f}%")
    axes[2].set_title("Volatility Trend Over Time", fontsize=12)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Volatility (%)")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    for ax in axes:
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_gpr(df))

with col2:
    st.pyplot(plot_garch(df))
