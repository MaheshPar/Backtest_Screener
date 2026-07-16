import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(
    page_title="Quantuous Momentum Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div.stButton > button:first-child {
        width: 100%;
        background-color: #1E1E24;
        color: white;
        border-radius: 6px;
        border: 1px solid #3A3A42;
        padding: 0.75rem;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #2D2D35;
        border-color: #4F4F5A;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SESSION STATE
# ==============================================================================
if "backtest_triggered" not in st.session_state:
    st.session_state.backtest_triggered = False

# ==============================================================================
# 3. STATIC DATA ENGINE & INDICATOR LOGIC
# ==============================================================================
@st.cache_data
def get_universe_tickers(universe_name: str) -> list:
    universes = {
        "NIFTY 50": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "BHARTIARTL.NS", "SBIN.NS", "LTIM.NS", "ITC.NS", "HINDUNILVR.NS",
            "LT.NS", "AXISBANK.NS", "ZOMATO.NS", "BAJFINANCE.NS", "MARUTI.NS",
            "TATASTEEL.NS", "M&M.NS", "SUNPHARMA.NS", "HCLTECH.NS", "KOTAKBANK.NS",
            "TITAN.NS", "ULTRACEMCO.NS", "ADANIENT.NS", "NTPC.NS", "POWERGRID.NS",
            "ASIANPAINT.NS", "JIOFIN.NS", "COALINDIA.NS", "INDUSINDBK.NS", "BAJAJFINSV.NS"
        ],
        "NIFTY 100": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "BHARTIARTL.NS", "SBIN.NS", "ZOMATO.NS", "HAL.NS", "BEL.NS",
            "TATAELXSI.NS", "TRENT.NS", "PFC.NS", "RECLTD.NS", "DMART.NS"
        ]
    }
    return universes.get(universe_name, universes["NIFTY 50"])

@st.cache_data(show_spinner=False)
def download_market_data(tickers: list, macro_ticker: str, period_years: int):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365 + 50)
    all_tickers = list(set(tickers + [macro_ticker]))
    
    try:
        raw_data = yf.download(
            tickers=all_tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            group_by='column',
            auto_adjust=True,
            progress=False
        )
        if raw_data.empty: return None, "Downloaded data frame is empty."
        return raw_data, None
    except Exception as e:
        return None, str(e)

# --- PURE VECTORIZED INDICATORS ---
def calc_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates RSI using pure vectorized operations across an entire pricing grid."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Using exponential moving average to match Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_technical_matrix(market_data: pd.DataFrame) -> dict:
    """Computes all required technical and momentum indicators simultaneously."""
    # Isolate closing prices
    close_prices = market_data['Close']
    
    # Absolute Momentum Filters
    ema_200 = close_prices.ewm(span=200, adjust=False).mean()
    rsi_14 = calc_rsi(close_prices, window=14)
    rolling_max_52w = close_prices.rolling(window=252, min_periods=1).max()
    dist_to_high = (close_prices / rolling_max_52w) - 1  # 0 to -X%
    
    # Relative Momentum Rankers
    daily_returns = close_prices.pct_change()
    abs_return_1yr = close_prices.pct_change(periods=252)
    
    # Annualized Sharpe Ratio (Mean of daily returns / Std Dev of daily returns) * sqrt(252)
    rolling_mean_ret = daily_returns.rolling(window=252).mean()
    rolling_std_ret = daily_returns.rolling(window=252).std()
    sharpe_1yr = (rolling_mean_ret / rolling_std_ret) * np.sqrt(252)
    
    return {
        "Close": close_prices,
        "EMA_200": ema_200,
        "RSI_14": rsi_14,
        "Dist_to_52W_High": dist_to_high,
        "Abs_Return_1yr": abs_return_1yr,
        "Sharpe_1yr": sharpe_1yr
    }

# ==============================================================================
# 4. TOP BRANDING & STATUS HEADER
# ==============================================================================
header_col1, header_col2 = st.columns([0.8, 0.2])
with header_col1:
    st.title("🎯 Quantuous")
    st.caption("Advanced Quantitative Screeners & Vectorized Backtesting Engines")
with header_col2:
    st.metric(label="Available Credits", value="11 cr", delta="Active Account")

st.divider()

# ==============================================================================
# 5. MAIN INTERFACE: CONFIGURATION PARAMETERS
# ==============================================================================
st.subheader("📊 Momentum Backtest Dashboard")

with st.form("backtest_parameters_form"):
    st.markdown("### 📁 Universe & Timing Setup")
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox("STOCK UNIVERSE", options=["NIFTY 50", "NIFTY 100"], index=0)
        backtest_period = st.select_slider("BACKTEST PERIOD", options=["2 Years", "5 Years", "10 Years"], value="5 Years")
    with col2:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            portfolio_size = st.number_input("PORTFOLIO SIZE (Max Assets)", min_value=1, max_value=50, value=10, step=1)
        with col_p2:
            rebalance_period = st.selectbox("REBALANCE PERIOD", options=["1 Month", "3 Months", "6 Months"], index=0)
        starting_capital = st.selectbox("STARTING CAPITAL (INR)", options=["1 Lac", "3 Lacs", "5 Lacs", "10 Lacs", "20 Lacs"], index=3)

    st.markdown("---")
    st.markdown("### ⚡ Momentum Setup Matrices")
    relative_momentum = st.selectbox("RELATIVE MOMENTUM CRITERIA", options=["Abs Return 1yr", "Sharpe 1yr"], index=0)
    
    st.markdown("#### **ABSOLUTE MOMENTUM CRITERIA**")
    grid1, grid2, grid3 = st.columns(3)
    with grid1: f_ema200 = st.checkbox("Close > EMA 200", value=True)
    with grid2: f_rsi60 = st.checkbox("RSI > 60", value=False)
    with grid3: f_high10 = st.checkbox("Within 10% of 52W High", value=False)

    st.markdown("---")
    st.markdown("### 🛡️ Macro Index & Safety Controls")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        nifty_filter = st.selectbox("NIFTY MARKET FILTER", options=["None", "NIFTY 50 > EMA 200"], index=1)
    with col_m2:
        max_rank_filter = st.selectbox("MAXIMUM RANK FILTER", options=["N/A"], index=0)

    submit_button = st.form_submit_button("🚀 Start Historical Backtest")

# ==============================================================================
# 6. PIPELINE EXECUTION
# ==============================================================================
if submit_button:
    st.session_state.backtest_triggered = True

if st.session_state.backtest_triggered:
    constituents = get_universe_tickers(universe)
    macro_index_ticker = "^NSEI"
    years_int = int(backtest_period.split()[0])
    
    with st.spinner("Downloading historical multi-asset OHLCV data..."):
        market_data, error_log = download_market_data(constituents, macro_index_ticker, years_int)
        
    if error_log:
        st.error(f"Data Pipeline Disrupted: {error_log}")
    else:
        # STEP 3 EXECUTION: Compute Technicals
        with st.spinner("Vectorizing Technical Data & Momentum Matrices..."):
            tech_matrices = compute_technical_matrix(market_data)
            
        st.success("Matrix calculations complete! All indicators vectorized successfully.")
        
        # Display the most recent technical snapshot for a few selected criteria
        with st.expander("🔬 View Current Technical State (Latest Trading Day)"):
            latest_close = tech_matrices["Close"].iloc[-1]
            latest_ema = tech_matrices["EMA_200"].iloc[-1]
            latest_rsi = tech_matrices["RSI_14"].iloc[-1]
            latest_1y_ret = tech_matrices["Abs_Return_1yr"].iloc[-1] * 100 # Convert to %
            
            snapshot_df = pd.DataFrame({
                "Close": latest_close,
                "EMA 200": latest_ema,
                "RSI 14": latest_rsi,
                "1Y Return (%)": latest_1y_ret,
                "Close > EMA 200?": latest_close > latest_ema
            }).dropna().round(2)
            
            # Exclude the macro index ticker from stock view
            if macro_index_ticker in snapshot_df.index:
                snapshot_df = snapshot_df.drop(macro_index_ticker)
                
            st.dataframe(snapshot_df.sort_values(by="1Y Return (%)", ascending=False), use_container_width=True)

        # Placeholders for Step 4
        st.markdown("### 📈 Performance Assessment Shell")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("CAGR", "-- %", "Waiting for Step 4")
        m_col2.metric("Max Drawdown", "-- %", "Waiting for Step 4")
        m_col3.metric("Sharpe Ratio", "--", "Waiting for Step 4")
        m_col4.metric("Win Rate", "-- %", "Waiting for Step 4")
