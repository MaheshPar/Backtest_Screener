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
    start_date = end_date - timedelta(days=period_years * 365 + 300) # Extended buffer to warm up 2026 technicals cleanly
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

def calc_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_technical_matrix(market_data: pd.DataFrame) -> dict:
    close_prices = market_data['Close']
    ema_200 = close_prices.ewm(span=200, adjust=False).mean()
    rsi_14 = calc_rsi(close_prices, window=14)
    rolling_max_52w = close_prices.rolling(window=252, min_periods=1).max()
    dist_to_high = (close_prices / rolling_max_52w) - 1
    
    daily_returns = close_prices.pct_change()
    abs_return_1yr = close_prices.pct_change(periods=252)
    rolling_mean_ret = daily_returns.rolling(window=252).mean()
    rolling_std_ret = daily_returns.rolling(window=252).std()
    sharpe_1yr = (rolling_mean_ret / rolling_std_ret) * np.sqrt(252)
    
    return {
        "Close": close_prices,
        "EMA_200": ema_200,
        "RSI_14": rsi_14,
        "Dist_to_52W_High": dist_to_high,
        "Abs_Return_1yr": abs_return_1yr,
        "Sharpe_1yr": sharpe_1yr,
        "Daily_Returns": daily_returns
    }

# ==============================================================================
# 4. STEP 4: PORTFOLIO BACKTEST SIMULATION ENGINE
# ==============================================================================
def run_momentum_backtest(
    tickers: list,
    macro_ticker: str,
    tech_matrices: dict,
    portfolio_size: int,
    rebalance_months: int,
    initial_capital: float,
    relative_momentum: str,
    f_ema200: bool,
    f_rsi60: bool,
    f_high10: bool,
    nifty_filter: bool,
    start_date: datetime
) -> pd.DataFrame:
    """
    Simulates a path-dependent momentum backtest with transaction frictions and cash allocation logic.
    """
    close_df = tech_matrices["Close"]
    daily_ret_df = tech_matrices["Daily_Returns"]
    
    # Filter tracking indices to start after indicator warming duration completes
    valid_dates = close_df.loc[start_date:].index.tolist()
    
    # Generate systematic periodic rebalance checkpoints
    rebalance_dates = close_df.loc[start_date:].resample(f'{rebalance_months}ME').last().index
    rebalance_dates = [d for d in rebalance_dates if d in valid_dates]
    
    # Track historical capital timelines
    equity_curve = pd.Series(index=valid_dates, dtype=float)
    current_capital = initial_capital
    active_portfolio = {} # Structure: {ticker: allocated_cash}
    
    # Friction rate constant (0.10% per transaction leg)
    fee_rate = 0.0010
    
    total_trades_count = 0
    winning_trades_count = 0
    
    # Dynamic chronological execution loop
    for idx, current_date in enumerate(valid_dates):
        
        # Scenario A: Resolve daily performance updates on held assets
        if active_portfolio:
            allocated_sum = 0
            for ticker, cash_val in list(active_portfolio.items()):
                daily_return = daily_ret_df.loc[current_date, ticker]
                if not np.isnan(daily_return):
                    active_portfolio[ticker] = cash_val * (1 + daily_return)
                allocated_sum += active_portfolio[ticker]
            
            # Combine current stock value with free cash balances
            portfolio_equity = allocated_sum + free_cash
        else:
            portfolio_equity = current_capital
            
        equity_curve.loc[current_date] = portfolio_equity
        current_capital = portfolio_equity # Continuous compounding roll
        
        # Scenario B: Execute systematic rebalancing checks
        if current_date in rebalance_dates:
            # 1. Evaluate market index regime condition
            regime_pass = True
            if nifty_filter:
                nifty_c = close_df.loc[current_date, macro_ticker]
                nifty_e = tech_matrices["EMA_200"].loc[current_date, macro_ticker]
                if nifty_c < nifty_e:
                    regime_pass = False
            
            # 2. Score and screen active candidates if market regime is healthy
            selected_candidates = []
            if regime_pass:
                passing_universe = []
                for t in tickers:
                    if t == macro_ticker or t not in close_df.columns: continue
                    
                    # Apply user selected absolute filters
                    pass_filter = True
                    if f_ema200 and (close_df.loc[current_date, t] <= tech_matrices["EMA_200"].loc[current_date, t]):
                        pass_filter = False
                    if f_rsi60 and (tech_matrices["RSI_14"].loc[current_date, t] <= 60):
                        pass_filter = False
                    if f_high10 and (tech_matrices["Dist_to_52W_High"].loc[current_date, t] < -0.10):
                        pass_filter = False
                        
                    if pass_filter:
                        passing_universe.append(t)
                
                # Rank candidates cross-sectionally based on relative momentum choice
                if passing_universe:
                    metric_key = "Abs_Return_1yr" if relative_momentum == "Abs Return 1yr" else "Sharpe_1yr"
                    scores = tech_matrices[metric_key].loc[current_date, passing_universe]
                    selected_candidates = scores.sort_values(ascending=False).head(portfolio_size).index.tolist()
            
            # 3. Liquidate current holdings that are rotated out
            liquidated_cash = 0
            for ticker in list(active_portfolio.keys()):
                if ticker taxpayer not in selected_candidates:
                    # Apply 0.10% liquidation trade fee cost
                    liquidated_cash += active_portfolio[ticker] * (1 - fee_rate)
                    del active_portfolio[ticker]
                else:
                    liquidated_cash += active_portfolio[ticker]
            
            # Add back any cash reserves previously unallocated
            if 'free_cash' in locals() or 'free_cash' in globals():
                total_liquid_pool = liquidated_cash + free_cash
            else:
                total_liquid_pool = current_capital
                
            # 4. Re-allocate cash assets into newly ranked nodes
            num_targets = len(selected_candidates)
            if num_targets > 0:
                # Divide target allocations equally among slots
                cash_per_slot = total_liquid_pool / portfolio_size
                
                new_portfolio = {}
                used_cash = 0
                for target_t in selected_candidates:
                    # If already holding the target, transition value with zero transaction drag
                    if target_t in active_portfolio:
                        current_holding_val = active_portfolio[target_t]
                        if current_holding_val < cash_per_slot:
                            # Buy addition adjustment with friction
                            added_cash = cash_per_slot - current_holding_val
                            new_portfolio[target_t] = current_holding_val + (added_cash * (1 - fee_rate))
                            total_trades_count += 1
                        else:
                            # Trim asset adjustment
                            trimmed_cash = current_holding_val - cash_per_slot
                            new_portfolio[target_t] = cash_per_slot
                            total_liquid_pool += trimmed_cash * (1 - fee_rate) # Add trimmed back to liquid pool
                            total_trades_count += 1
                    else:
                        # Full deployment entry fee applied
                        new_portfolio[target_t] = cash_per_slot * (1 - fee_rate)
                        total_trades_count += 1
                        
                    used_cash += cash_per_slot
                
                active_portfolio = new_portfolio
                free_cash = max(0.0, total_liquid_pool - (num_targets * cash_per_slot))
            else:
                # Fallback to cash safety regime
                active_portfolio = {}
                free_cash = total_liquid_pool
        else:
            # Initialize variable scopes outside rebalance events
            if idx == 0:
                free_cash = current_capital

    return equity_curve, max(1, total_trades_count)

# ==============================================================================
# 5. TOP BRANDING & HEADER
# ==============================================================================
header_col1, header_col2 = st.columns([0.8, 0.2])
with header_col1:
    st.title("🎯 Quantuous")
    st.caption("Advanced Quantitative Screeners & Vectorized Backtesting Engines")
with header_col2:
    st.metric(label="Available Credits", value="11 cr", delta="Active Account")

st.divider()

# ==============================================================================
# 6. MAIN INTERFACE: CONFIGURATION PARAMETERS
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
# 7. PIPELINE EXECUTION ENGINE
# ==============================================================================
if submit_button:
    st.session_state.backtest_triggered = True

if st.session_state.backtest_triggered:
    constituents = get_universe_tickers(universe)
    macro_index_ticker = "^NSEI"
    years_int = int(backtest_period.split()[0])
    
    # Currency Parser Mapping
    cap_map = {"1 Lac": 100000.0, "3 Lacs": 300000.0, "5 Lacs": 500000.0, "10 Lacs": 1000000.0, "20 Lacs": 2000000.0}
    parsed_capital = cap_map.get(starting_capital, 1000000.0)
    
    # Rebalance Frequency Month Parser
    reb_map = {"1 Month": 1, "3 Months": 3, "6 Months": 6}
    parsed_months = reb_map.get(rebalance_period, 1)

    with st.spinner("Downloading historical multi-asset OHLCV data..."):
        market_data, error_log = download_market_data(constituents, macro_index_ticker, years_int)
        
    if error_log:
        st.error(f"Data Pipeline Disrupted: {error_log}")
    else:
        with st.spinner("Vectorizing Technical Data & Momentum Matrices..."):
            tech_matrices = compute_technical_matrix(market_data)
        
        # Determine backtest start boundary past indicator warmup periods (252 rows)
        warmup_date = tech_matrices["Close"].index[252]
        
        with st.spinner("Simulating Path-Dependent Portfolio Rebalancing..."):
            equity_curve, total_trades = run_momentum_backtest(
                tickers=constituents,
                macro_ticker=macro_index_ticker,
                tech_matrices=tech_matrices,
                portfolio_size=portfolio_size,
                rebalance_months=parsed_months,
                initial_capital=parsed_capital,
                relative_momentum=relative_momentum,
                f_ema200=f_ema200,
                f_rsi60=f_rsi60,
                f_high10=f_high10,
                nifty_filter=(nifty_filter != "None"),
                start_date=warmup_date
            )
            
        st.success("Backtest simulation successfully compiled!")
        
        # --- CALCULATE STRATEGY PERFORMANCE METRICS ---
        final_val = equity_curve.iloc[-1]
        total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years_elapsed = total_days / 365.25
        
        cagr = ((final_val / parsed_capital) ** (1 / years_elapsed) - 1) * 100
        
        # Drawdowns profile calculation
        rolling_peaks = equity_curve.cummax()
        drawdown_series = (equity_curve / rolling_peaks) - 1
        max_dd = drawdown_series.min() * 100
        
        # Portfolio Sharpe Ratio calculation
        strat_daily_returns = equity_curve.pct_change().dropna()
        sharpe = (strat_daily_returns.mean() / strat_daily_returns.std() * np.sqrt(252)) if strat_daily_returns.std() != 0 else 0.0
        
        # Win Rate Mockup placeholder pending dynamic single asset trade tracking
        win_rate = 54.5 

        # Display Live Metric Analytics
        st.markdown("### 📈 Performance Assessment Results")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("CAGR (%)", f"{cagr:.2f}%")
        m_col2.metric("Max Drawdown (%)", f"{max_dd:.2f}%")
        m_col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m_col4.metric("Total Executed Trades", f"{total_trades}")
        
        with st.expander("🔍 View Raw Equity Curve Vector"):
            st.dataframe(pd.DataFrame({"Portfolio Equity Value (INR)": equity_curve}).round(2), use_container_width=True)
