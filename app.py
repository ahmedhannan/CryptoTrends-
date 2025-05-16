import os
import streamlit as st
import io
import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import main_agent
from portfolio_optimizer import PortfolioOptimizer
from config import INITIAL_CAPITAL, TRANSACTION_COST_PCT

NEWSAPI_KEY = 'a3421e970ed4469085e0e99731010dc6'#'dc350ab9fe2c4b44906c8dad314384f4'  # os.getenv("NEWSAPI_KEY", "")

st.set_page_config(
    page_title="CryptoAgent Dashboard",
    layout="wide",
)

def get_live_ohlcv(symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    mapping = {
        'bitcoin': 'BTCUSDT',
        'ethereum': 'ETHUSDT',
        'ripple': 'XRPUSDT',
        'binancecoin': 'BNBUSDT',
        'solana': 'SOLUSDT',
    }
    market = mapping.get(symbol.lower())
    if not market:
        raise ValueError(f"No ticker for symbol '{symbol}'")
    url = f"https://api.binance.com/api/v3/klines?symbol={market}&interval={interval}&limit={limit}"
    data = requests.get(url, timeout=10).json()
    df = pd.DataFrame(data, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_asset_volume','num_trades',
        'taker_buy_base','taker_buy_quote','ignore'
    ])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.rename(columns={
        'open':'open_price',
        'high':'high_price',
        'low':'low_price',
        'close':'close_price'
    })
    df.set_index('timestamp', inplace=True)
    return df[['open_price','high_price','low_price','close_price','volume']]

# --- Fetch news ---
def fetch_news(symbol: str, start: datetime, end: datetime):
    if not NEWSAPI_KEY:
        return []
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': symbol,
        'from': start.isoformat(),
        'to': end.isoformat(),
        'sortBy': 'relevancy',
        'language': 'en',
        'pageSize': 3,
        'apiKey': NEWSAPI_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return []
    return [{'title': a['title'], 'url': a['url']} for a in res.json().get('articles', [])]

@st.cache_resource
def load_forecasting_artifacts(symbol: str):
    return main_agent.load_forecasting_model(symbol)

@st.cache_data(show_spinner=False)
def forecast(symbol: str, hours: int) -> pd.DataFrame:
    model, scalers, features = load_forecasting_artifacts(symbol)
    df_live = get_live_ohlcv(symbol, limit=200)
    df_feat = main_agent.add_technical_indicators(df_live)
    window = df_feat[features].iloc[-main_agent.SEQUENCE_LENGTH:]
    df_pred = main_agent.predict_future_prices(
        keras_model=model,
        data_scalers_dict=scalers,
        model_input_features_list=features,
        recent_ohlcv_featured_df=window,
        main_target_feature_name='close_price'
    )
    if 'timestamp' in df_pred.columns:
        df_pred.index = pd.to_datetime(df_pred['timestamp'])
    return df_pred.head(hours)

@st.cache_data(show_spinner=False)
def detect_anomalies(symbol: str):
    model, scaler, threshold = main_agent.load_anomaly_detector_components(symbol)
    df_live = get_live_ohlcv(symbol, limit=200)
    df_feat = main_agent.add_technical_indicators(df_live)
    window = df_feat.select_dtypes(include=[float, int]).iloc[-main_agent.SEQUENCE_LENGTH:]
    df_anom = main_agent.detect_anomalies_with_autoencoder(model, scaler, threshold, window)
    multiplier = 1.5
    filtered = df_anom[df_anom['anomaly_reconstruction_error'] > threshold * multiplier]
    if 'timestamp' in df_anom.columns:
        filtered.index = pd.to_datetime(filtered['timestamp'])
    return filtered, threshold, df_live

@st.cache_data(show_spinner=False)
def optimize_portfolio():
    symbols = main_agent.TARGET_CRYPTOS
    pct_changes = {}
    for sym in symbols:
        df_live = get_live_ohlcv(sym, limit=200)
        past = df_live['close_price'].tail(96)
        if len(past) < 2 or past.iloc[0] == 0:
            pct = 0.0
        else:
            pct = (past.iloc[-1] - past.iloc[0]) / past.iloc[0]
        pct_changes[sym] = pct
    investments = {}
    total_positive = sum(p for p in pct_changes.values() if p > 0)
    for sym, p in pct_changes.items():
        if p > 0 and total_positive > 0:
            weight = p / total_positive
            investments[sym] = weight * INITIAL_CAPITAL
        else:
            investments[sym] = 0.0
    df_inv = pd.DataFrame({
        'symbol': list(investments.keys()),
        'investment': list(investments.values()),
        'momentum_pct': [pct_changes[s] for s in investments.keys()]
    })
    return df_inv

# --- LAYOUT ---
st.title('📊 CryptoAgent Dashboard')
st.caption(f'Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}')
tabs = st.tabs(['Forecasting','Anomaly','Portfolio'])

# Forecasting
with tabs[0]:
    st.header('🔮 Price Forecasting')
    sym = st.selectbox('Crypto', main_agent.TARGET_CRYPTOS)
    hrs = st.slider('Hours Ahead', 1, main_agent.PREDICTION_HORIZON, 6)
    if st.button('Run Forecast'):
        dfp = forecast(sym, hrs)
        df_live = get_live_ohlcv(sym, limit=200)
        if dfp.empty:
            st.warning('No forecast data.')
        else:
            pred_col = next((c for c in dfp.columns if c.startswith('predicted_')), dfp.columns[0])
            pred_time = dfp.index[0]
            pred_val = dfp[pred_col].iloc[0]
            st.metric(f"{hrs}h Forecast", f"${pred_val:.2f}")
            # Show past 24 hours trend
            past = df_live['close_price'].tail(24)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(past.index, past.values, label='Past 24h Price', color='grey', alpha=0.6)
            ax.scatter([pred_time], [pred_val], color='red', s=100, label='Predicted Price')
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            fig.autofmt_xdate()
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

# Anomaly Detection
with tabs[1]:
    st.header('🚨 Anomaly Detection')
    sym2 = st.selectbox('Crypto', main_agent.TARGET_CRYPTOS, key='anom2')
    if st.button('Detect Anomalies'):
        filtered, thresh, live = detect_anomalies(sym2)
        st.markdown(f"**Threshold:** {thresh:.4f}  (multiplier 1.5)")
        if filtered.empty:
            st.success('No significant anomalies.')
        else:
            st.subheader(f'Anomalies for {sym2}')
            st.dataframe(filtered)
            st.metric('Count', len(filtered))
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(live['close_price'], label='Price')
            ax.scatter(filtered.index, live.loc[filtered.index,'close_price'], color='red', label='Anomaly')
            ax.legend()
            st.pyplot(fig)
            st.subheader('Contextual News')
            for ts in filtered.index:
                arts = fetch_news(sym2, ts - timedelta(hours=1), ts + timedelta(hours=1))
                st.markdown(f"**Anomaly at {ts}**")
                if arts:
                    for a in arts:
                        st.markdown(f"- [{a['title']}]({a['url']})")
                else:
                    st.markdown('_No news found or missing API key_')

# Portfolio
with tabs[2]:
    st.header('💼 Portfolio Optimization')
    if st.button('Optimize Portfolio'):
        df_inv = optimize_portfolio()
        st.subheader('Investment Suggestions')
        for _, row in df_inv.iterrows():
            sym = row['symbol']
            amt = row['investment']
            if amt > 0:
                st.markdown(f"✅ Invest **${amt:,.2f}** in **{sym}** (Momentum: {row['momentum_pct']*100:.1f}%)")
            else:
                st.markdown(f"❌ Do not invest in **{sym}** (Momentum: {row['momentum_pct']*100:.1f}%)")
        fig, ax = plt.subplots(figsize=(8,4))
        colors = ['green' if x>0 else 'grey' for x in df_inv['investment']]
        ax.bar(df_inv['symbol'], df_inv['investment'], color=colors)
        ax.set_ylabel('Suggested Investment ($)')
        ax.set_xlabel('Cryptocurrency')
        ax.set_title('Investment Allocation Based on past 4 days Momentum')
        st.pyplot(fig)

# Logs
st.sidebar.header('📜 Logs')
if st.sidebar.button('Run'):
    buf, real = io.StringIO(), sys.stdout; sys.stdout = buf
    main_agent.run_agent_orchestration_cycle(); sys.stdout = real
    st.sidebar.text_area('Logs', buf.getvalue(), height=300)

# Footer
st.markdown('---')
st.caption('© 2025 CryptoAgent')
