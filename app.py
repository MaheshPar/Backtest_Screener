import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME ATTEMPTS
# ==============================================================================
st.set_page_config(
    page_title="Quantuous Momentum Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom minimal CSS for slate layout tones, grid structures, and tight margins
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
""", unsafe_style_html=True)

# ==============================================================================
# 2. SESSION STATE INITIALIZATION
# ==============================================================================
if "backtest_triggered" not in st.session_state:
    st.session_state.backtest_triggered = False

# ==============================================================================
# 3. STATIC UNIVERSE MAPPINGS & CACHED DATA ENGINE
# ==============================================================================
@st.cache_data
def get_universe_tickers(universe_name: str) -> list:
    """
    Returns the constituent ticker list for the selected Indian index.
    In production, this can parse live CSV targets from the NSE website.
    To ensure out-of-the-box reliability, a high-conviction structural sample is mapped.
    """
    # Base dictionaries mapping Indian equity tickers (.NS suffix)
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
    
    # Fallback to NIFTY 50 if the sample universe isn't explicitly defined in this mockup
    return universes.get(universe_name, universes["NIFTY 50"])

@st.cache_data(show_spinner=False)
def download_market_data(tickers: list, macro_ticker: str, period_years: int):
    """
    Downloads historical adjusted OHLCV data from yfinance with optimized parallel caching.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_years * 365 + 50) # Buffer added for moving average initialization
    
    all_tickers = list(set(tickers + [macro_ticker]))
    
    try:
        # Fetching adjusted OHLCV data
        raw_data = yf.download(
            tickers=all_tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            group_by='column',
            auto_adjust=True, # Critical for accurate momentum calculations without artificial corporate splits bias
            progress=False
        )
        
        if raw_data.empty:
            return None, "Downloaded data frame is completely empty."
            
        return raw_data, None
    except Exception as e:
        return None, str(e)

# ==============================================================================
# 4. TOP BRANDING & STATUS HEADER
# ==============================================================================
header_col1, header_col2 = st.columns([0.8, 0.2])
with header_col1:
    st.title("🎯 Quantuous")
    st.caption("Advanced Quantitative Screeners & Vectorized Backtesting Engines")
with header_col2:
    st.metric(label="Available Credits", value="11 cr", delta="Active Account")

st.hr()

# ==============================================================================
# 5. MAIN INTERFACE: CONFIGURATION PARAMETERS
# ==============================================================================
st.subheader("📊 Momentum Backtest Dashboard")
st.caption("Configure asset universes, timing profiles, and multi-tiered filtering rules below.")

with st.form("backtest_parameters_form"):
    
    # --- SECTION A: STOCK UNIVERSE & TIMING CONFIGURATION ---
    st.markdown("### 📁 Universe & Timing Setup")
    col1, col2 = st.columns(2)
    
    with col1:
        universe = st.selectbox(
            "STOCK UNIVERSE",
            options=["NIFTY 50", "NIFTY 100"],
            index=0,
            help="The structural asset pool from which momentum candidates will be sourced."
        )
        
        backtest_period = st.select_slider(
            "BACKTEST PERIOD",
            options=["2 Years", "5 Years", "10 Years"],
            value="5 Years",
            help="Historical testing lookback window duration."
        )

    with col2:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            portfolio_size = st.number_input(
                "PORTFOLIO SIZE (Max Assets)",
                min_value=1, max_value=50, value=10, step=1,
                help="Maximum allocation nodes allowed concurrently inside the portfolio."
            )
        with col_p2:
            rebalance_period = st.selectbox(
                "REBALANCE PERIOD",
                options=["1 Month", "3 Months", "6 Months"],
                index=0,
                help="Cadence intervals where rankings are reassessed and trades executed."
            )
            
        starting_capital = st.selectbox(
            "STARTING CAPITAL (INR)",
            options=["1 Lac", "3 Lacs", "5 Lacs", "10 Lacs", "20 Lacs"],
            index=3,
            help="Initial principal cash deployment value for capital allocation metrics."
        )

    st.markdown("---")

    # --- SECTION B: MOMENTUM RANKING CRITERIA ---
    st.markdown("### ⚡ Momentum Setup Matrices")
    
    relative_momentum = st.selectbox(
        "RELATIVE MOMENTUM CRITERIA (Cross-sectional Ranking)",
        options=["Abs Return 1yr", "Sharpe 1yr"],
        index=0,
        help="Metric variant utilized to score assets cross-sectionally for priority selection."
    )
    
    st.markdown("#### **ABSOLUTE MOMENTUM CRITERIA (Binary Condition Filters)**")
    
    grid1, grid2, grid3 = st.columns(3)
    with grid1:
        st.markdown("**Trend & Trajectory**")
        f_ema200 = st.checkbox("Close > EMA 200", value=True, help="Classic institutional long-term bias filter.")
    with grid2:
        st.markdown("**Oscillators & Volatility**")
        f_rsi60 = st.checkbox("RSI > 60", value=False)
    with grid3:
        st.markdown("**Price Action Execution**")
        f_high10 = st.checkbox("Within 10% of 52W High", value=False)

    st.markdown("---")

    # --- SECTION C: MACRO / RISK MANAGERS ---
    st.markdown("### 🛡️ Macro Index & Safety Controls")
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        nifty_filter = st.selectbox(
            "NIFTY MARKET FILTER (Regime Filter)",
            options=["None", "NIFTY 50 > EMA 200"],
            index=1,
            help="Beta market regime filter. Goes to cash or stops entries if the index breaks below this floor."
        )
    with col_m2:
        max_rank_filter = st.selectbox(
            "MAXIMUM RANK FILTER",
            options=["N/A"],
            index=0
        )

    submit_button = st.form_submit_button("🚀 Start Historical Backtest")

# ==============================================================================
# 6. BACKTEST TRIGGER HANDLING & DATA INGESTION
# ==============================================================================
if submit_button:
    st.session_state.backtest_triggered = True

if st.session_state.backtest_triggered:
    st.success(f"Parameters locked! Resolving tickers for {universe}...")
    
    # 1. Resolve Universe Constituents
    constituents = get_universe_tickers(universe)
    macro_index_ticker = "^NSEI" # Nifty 50 Index for absolute regime filter tracking
    
    st.info(f"Identified {len(constituents)} index assets. Commencing data ingestion pipeline via yfinance...")
    
    # 2. Extract Timing Window Numerical Value
    years_int = int(backtest_period.split()[0])
    
    # 3. Pull Cached Asset Histories
    with st.spinner("Downloading historical multi-asset OHLCV data frames..."):
        market_data, error_log = download_market_data(
            tickers=constituents, 
            macro_ticker=macro_index_ticker, 
            period_years=years_int
        )
        
    if error_log:
        st.error(f"Data Pipeline Disrupted: {error_log}")
    else:
        st.success("Historical data arrays successfully downlinked and cached!")
        
        # Display operational telemetry regarding shape of downloaded data matrix
        with st.expander("🔍 Inspect Ingested Market Data Matrix Structures"):
            st.write("Raw multi-index column schema details:")
            st.dataframe(market_data.head(5))
            st.caption(f"Data Matrix Dimension: {market_data.shape[0]} trading rows across {market_data.shape[1]} series layers.")
            
        # Visual placeholder container blocks representing upcoming processing engines
        st.markdown("### 📈 Performance Assessment Shell")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("CAGR", "-- %", "Waiting for Step 4")
        m_col2.metric("Max Drawdown", "-- %", "Waiting for Step 4")
        m_col3.metric("Sharpe Ratio", "--", "Waiting for Step 4")
        m_col4.metric("Win Rate", "-- %", "Waiting for Step 4")
