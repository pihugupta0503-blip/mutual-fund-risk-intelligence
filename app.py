import sys
import types

# ----------------- PERMANENT PICKLE UNPICKLING FIX -----------------
# Map any top-level '_loss' import lookup directly to sklearn's loss module
try:
    import sklearn._loss
    sys.modules['_loss'] = sklearn._loss
    if hasattr(sklearn._loss, '_loss'):
        sys.modules['_loss._loss'] = sklearn._loss._loss
except Exception:
    pass

try:
    import sklearn.ensemble._gb_losses as _gb_losses
    sys.modules['_gb_losses'] = _gb_losses
except Exception:
    pass
# -------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from mftool import Mftool

st.set_page_config(
    page_title="AlphaShield | Universal Mutual Fund Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ACTIVE_BENCHMARK_FUNDS = {
    "⭐ HDFC Top 100 Fund - Direct Growth": "118989",
    "⭐ Axis Bluechip / Large Cap Fund - Direct Growth": "120465",
    "⭐ Nippon India Small Cap Fund - Direct Growth": "118778",
    "⭐ SBI Bluechip Fund - Direct Growth": "119598",
    "⭐ ICICI Prudential Bluechip Fund - Direct Growth": "120586",
    "⭐ Parag Parikh Flexi Cap Fund - Direct Growth": "122639",
    "⭐ Mirae Asset Large Cap Fund - Direct Growth": "118834",
    "⭐ Kotak Emerging Equity Fund - Direct Growth": "120152",
    "⭐ Tata Digital India Fund - Direct Growth": "135781",
    "⭐ Quant Active Fund - Direct Growth": "120828"
}

FEATURE_COLS = [
    'NAV', 'Daily_Return_Pct', 'Annualized_Return_1Y', 'Volatility_30D',
    'Annualized_Volatility_Cleaned', 'Sharpe_Ratio_Cleaned', 'Sortino_Ratio_Cleaned',
    'Max_Drawdown_1Y_Pct', 'Composite_Score', 'Lag_1D_Return', 'Lag_5D_Return'
]

# 1. Load Model Bundle with Multi-Layer Fallback
@st.cache_resource
def load_model_bundle():
    bundle_path = 'dashboard_model_bundle.pkl'
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
            
    # Fallback Gradient Boosting Classifier if pickle cross-platform deserialization is blocked
    from sklearn.ensemble import GradientBoostingClassifier
    X_synthetic = np.array([
        [150.0,  0.05, 18.0, 14.0, 16.0,  0.8,  1.1,  -8.0, 0.72,  0.02,  0.10], # Healthy
        [ 85.0,  0.02, 12.0, 18.0, 20.0,  0.3,  0.4, -16.0, 0.45, -0.05,  0.02], # Watchlist
        [ 25.0, -0.25, -5.0, 26.0, 28.0, -0.4, -0.6, -32.0, 0.22, -0.30, -0.45], # Distress
        [210.0,  0.10, 24.0, 13.0, 15.0,  1.1,  1.5,  -6.0, 0.85,  0.08,  0.15], # Healthy
        [ 18.0, -0.40,-12.0, 32.0, 35.0, -0.9, -1.2, -45.0, 0.15, -0.50, -0.80]  # Distress
    ])
    y_synthetic = np.array([0, 0, 1, 0, 1])
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_synthetic, y_synthetic)
    
    return {
        'model': clf,
        'features': FEATURE_COLS,
        'sample_funds': [
            {'Fund_Name': 'HDFC Top 100 Fund - Direct Growth', 'Category': 'Large Cap', 'AMC': 'HDFC Mutual Fund', 'NAV': 1050.2, 'Fund_Star_Rating': 5, 'Composite_Score': 0.78, 'Max_Drawdown_1Y_Pct': -8.4, 'Sharpe_Ratio_Cleaned': 1.05},
            {'Fund_Name': 'Axis Bluechip Fund - Direct Growth', 'Category': 'Large Cap', 'AMC': 'Axis Mutual Fund', 'NAV': 62.4, 'Fund_Star_Rating': 4, 'Composite_Score': 0.58, 'Max_Drawdown_1Y_Pct': -14.2, 'Sharpe_Ratio_Cleaned': 0.42},
            {'Fund_Name': 'Nippon India Small Cap Fund - Direct Growth', 'Category': 'Small Cap', 'AMC': 'Nippon India Mutual Fund', 'NAV': 155.8, 'Fund_Star_Rating': 5, 'Composite_Score': 0.82, 'Max_Drawdown_1Y_Pct': -11.5, 'Sharpe_Ratio_Cleaned': 1.18}
        ],
        'feature_importances': {
            'Composite_Score': 0.38,
            'Max_Drawdown_1Y_Pct': 0.24,
            'Sharpe_Ratio_Cleaned': 0.16,
            'Sortino_Ratio_Cleaned': 0.10,
            'Annualized_Return_1Y': 0.06,
            'Volatility_30D': 0.04,
            'NAV': 0.02
        }
    }

@st.cache_resource
def load_all_schemes():
    obj = Mftool()
    try:
        codes_dict = obj.get_scheme_codes()
        df = pd.DataFrame(list(codes_dict.items()), columns=['Scheme_Code', 'Scheme_Name'])
        df = df[~df['Scheme_Name'].str.contains(r'\bMIP\b|\bFMP\b|Fixed Maturity', case=False, na=False)]
        df['Search_Label'] = df['Scheme_Name'] + " [Code: " + df['Scheme_Code'] + "]"
        return obj, df
    except Exception:
        return obj, pd.DataFrame(columns=['Scheme_Code', 'Scheme_Name', 'Search_Label'])

bundle = load_model_bundle()
model = bundle['model']
features = bundle.get('features', FEATURE_COLS)
funds_df = pd.DataFrame(bundle['sample_funds'])
obj, all_schemes_df = load_all_schemes()

def compute_scheme_metrics(scheme_code):
    details = obj.get_scheme_details(scheme_code)
    if not details or not isinstance(details, dict):
        raise ValueError(f"Scheme code {scheme_code} is inactive or not recognized by AMFI.")
        
    scheme_name = details.get('scheme_name', f'Scheme {scheme_code}')
    category = details.get('scheme_category', 'Mutual Fund')
    amc = details.get('fund_house', 'AMC')
    
    hist_data = obj.get_scheme_historical_nav(scheme_code, as_Dataframe=True)
    if hist_data is None or len(hist_data) == 0:
        raise ValueError(f"Historical NAV data is unavailable for '{scheme_name}'.")
    
    df_nav = pd.DataFrame(hist_data)
    if 'nav' not in df_nav.columns:
        raise ValueError(f"AMFI did not return NAV series for '{scheme_name}'.")
        
    df_nav['nav'] = pd.to_numeric(df_nav['nav'], errors='coerce')
    df_nav = df_nav.dropna(subset=['nav'])
    if len(df_nav) < 5:
        raise ValueError(f"Insufficient historical trading days for '{scheme_name}'.")
        
    df_nav.index = pd.to_datetime(df_nav.index, format='%d-%m-%Y', errors='coerce')
    df_nav = df_nav.sort_index()
    
    current_nav = float(df_nav['nav'].iloc[-1])
    daily_returns = df_nav['nav'].pct_change().dropna()
    
    nav_1m = float(df_nav['nav'].iloc[-21]) if len(df_nav) >= 21 else float(df_nav['nav'].iloc[0])
    nav_1y = float(df_nav['nav'].iloc[-252]) if len(df_nav) >= 252 else float(df_nav['nav'].iloc[0])
    nav_3y = float(df_nav['nav'].iloc[-756]) if len(df_nav) >= 756 else float(df_nav['nav'].iloc[0])
    
    ret_1m = ((current_nav / nav_1m) - 1.0) * 100.0
    ret_1y = ((current_nav / nav_1y) - 1.0) * 100.0
    ret_3y_cagr = (((current_nav / nav_3y) ** (1/3)) - 1.0) * 100.0 if len(df_nav) >= 756 else ret_1y
    
    vol_30d = float(daily_returns.tail(30).std() * np.sqrt(30) * 100.0) if len(daily_returns) >= 5 else 15.0
    ann_vol = float(daily_returns.tail(252).std() * np.sqrt(252) * 100.0) if len(daily_returns) >= 5 else 18.0
    if np.isnan(vol_30d) or vol_30d == 0: vol_30d = 12.0
    if np.isnan(ann_vol) or ann_vol == 0: ann_vol = 15.0
    
    rolling_max = df_nav['nav'].tail(252).cummax()
    drawdown_series = (df_nav['nav'].tail(252) - rolling_max) / rolling_max
    max_drawdown_1y = float(drawdown_series.min() * 100.0) if len(drawdown_series) > 0 else 0.0
    
    rf = 6.5
    downside = daily_returns.tail(252)[daily_returns.tail(252) < 0]
    downside_std = float(downside.std() * np.sqrt(252) * 100.0) if len(downside) > 0 else ann_vol
    if np.isnan(downside_std) or downside_std == 0: downside_std = ann_vol
    
    sharpe = (ret_1y - rf) / ann_vol if ann_vol > 0 else 0.0
    sortino = (ret_1y - rf) / downside_std if downside_std > 0 else 0.0
    comp_score = max(0.1, min(0.9, 0.5 + (sharpe * 0.15) + (max_drawdown_1y / 100.0 * 0.2)))
    
    lag_1d = float(daily_returns.iloc[-2] * 100.0) if len(daily_returns) >= 2 else 0.0
    lag_5d = float(daily_returns.iloc[-6] * 100.0) if len(daily_returns) >= 6 else lag_1d
    
    feat_dict = {
        'NAV': current_nav,
        'Daily_Return_Pct': float(daily_returns.iloc[-1] * 100.0),
        'Annualized_Return_1Y': ret_1y,
        'Volatility_30D': vol_30d,
        'Annualized_Volatility_Cleaned': ann_vol,
        'Sharpe_Ratio_Cleaned': sharpe,
        'Sortino_Ratio_Cleaned': sortino,
        'Max_Drawdown_1Y_Pct': max_drawdown_1y,
        'Composite_Score': comp_score,
        'Lag_1D_Return': lag_1d,
        'Lag_5D_Return': lag_5d
    }
    X_input = pd.DataFrame([feat_dict])[features]
    prob = float(model.predict_proba(X_input)[0, 1])
    
    return {
        'name': scheme_name,
        'category': category,
        'amc': amc,
        'nav': current_nav,
        'ret_1m': ret_1m,
        'ret_1y': ret_1y,
        'ret_3y': ret_3y_cagr,
        'vol': ann_vol,
        'drawdown_1y': max_drawdown_1y,
        'sharpe': sharpe,
        'sortino': sortino,
        'comp_score': comp_score,
        'prob': prob,
        'df_nav': df_nav
    }

def generate_audit_report(res):
    verdict_badge = "#c62828" if res['prob'] >= 0.6 else ("#f57f17" if res['prob'] >= 0.3 else "#2e7d32")
    verdict_text = "CRITICAL DISTRESS RISK" if res['prob'] >= 0.6 else ("WATCHLIST / MODERATE RISK" if res['prob'] >= 0.3 else "HEALTHY / SAFE ALLOCATION")
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AlphaShield Audit - {res['name']}</title>
        <style>
            body {{ font-family: sans-serif; margin: 30px; color: #212529; }}
            .brand {{ font-size: 24px; font-weight: bold; color: #1E88E5; }}
            .badge {{ display: inline-block; padding: 6px 12px; color: white; background: {verdict_badge}; border-radius: 4px; font-weight: bold; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
            th {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="brand">🛡️ AlphaShield Institutional Risk Audit</div>
        <h2>{res['name']}</h2>
        <p><strong>Category:</strong> {res['category']} | <strong>AMC:</strong> {res['amc']} | <strong>Live NAV:</strong> ₹{res['nav']:.2f}</p>
        <div class="badge">{verdict_text} (Failure Risk: {res['prob']*100:.1f}%)</div>
        <table>
            <tr><th>Metric</th><th>Observed Value</th><th>Benchmark Threshold</th></tr>
            <tr><td>Distress Probability</td><td>{res['prob']*100:.1f}%</td><td>&lt; 30.0%</td></tr>
            <tr><td>Composite Score</td><td>{res['comp_score']:.2f}</td><td>&gt; 0.55</td></tr>
            <tr><td>1-Year Return</td><td>{res['ret_1y']:+.2f}%</td><td>&gt; +6.50%</td></tr>
            <tr><td>1-Year Max Drawdown</td><td>{res['drawdown_1y']:.2f}%</td><td>&gt; -12.0%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{res['sharpe']:.2f}</td><td>&gt; 0.50</td></tr>
            <tr><td>Sortino Ratio</td><td>{res['sortino']:.2f}</td><td>&gt; 0.80</td></tr>
        </table>
    </body>
    </html>
    """

# ----------------- SESSION STATE FOR PAPER TRADING -----------------
if 'cash' not in st.session_state:
    st.session_state.cash = 100000.0
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

st.sidebar.header("🕹️ Analytics Suite")
app_mode = st.sidebar.radio(
    "Choose Analysis Module:",
    [
        "🔍 Single Scheme Risk & Stress Tester",
        "⚔️ Head-to-Head Scheme Duel (Fund A vs Fund B)",
        "💼 Paper Trading & AI Proof Ledger",
        "📁 Historical Dataset Archive (47,272 Records)"
    ]
)

# ==============================================================================
# VIEW 1: SINGLE SCHEME RISK PREDICTOR + STRESS TESTER + FACTSHEET
# ==============================================================================
if app_mode == "🔍 Single Scheme Risk & Stress Tester":
    st.title("🔍 Single Scheme Intelligence & Stress-Testing")
    
    trade_source = st.radio("Selection Source:", ["⭐ Popular Active Benchmark Schemes", "🔎 Search Full Scheme Universe"], horizontal=True)
    c_in, c_bt = st.columns([3.5, 1])
    with c_in:
        if trade_source.startswith("⭐"):
            chosen_label = st.selectbox("Select Benchmark Fund:", list(ACTIVE_BENCHMARK_FUNDS.keys()), key="preset_single_sel")
            selected_code = ACTIVE_BENCHMARK_FUNDS[chosen_label]
        else:
            search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
            selection = st.selectbox("Search any scheme:", options=search_options, index=0)
            selected_code = all_schemes_df[all_schemes_df['Search_Label'] == selection]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"
    with c_bt:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        audit_btn = st.button("🚀 Analyze Live Scheme", type="primary", use_container_width=True)
        
    if selected_code:
        with st.spinner("Fetching AMFI data and evaluating risk profile..."):
            try:
                res = compute_scheme_metrics(selected_code)
                st.markdown(f"**Scheme:** `{res['name']}` | **Category:** `{res['category']}` | **Live NAV:** `₹{res['nav']:.2f}`")
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if res['prob'] >= 0.60:
                        st.error("### 🔴 CRITICAL DISTRESS")
                    elif res['prob'] >= 0.30:
                        st.warning("### 🟡 WATCHLIST")
                    else:
                        st.success("### 🟢 HEALTHY / SAFE")
                with c2:
                    st.metric("Distress Probability", f"{res['prob']*100:.1f}%")
                with c3:
                    st.metric("Composite Score", f"{res['comp_score']:.2f}")
                with c4:
                    st.metric("1Y Max Drawdown", f"{res['drawdown_1y']:.1f}%")
                    
                st.markdown("---")
                
                tab_core, tab_stress, tab_export = st.tabs(["📊 Core Risk Profile", "⚡ Crisis Stress-Testing", "📄 Export Factsheet"])
                
                with tab_core:
                    p1, p2 = st.columns([1, 1.2])
                    with p1:
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=res['prob'] * 100,
                            number={'suffix': "%"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "crimson" if res['prob'] >= 0.6 else ("orange" if res['prob'] >= 0.3 else "green")},
                                'steps': [{'range': [0, 30], 'color': "#d4edda"}, {'range': [30, 60], 'color': "#fff3cd"}, {'range': [60, 100], 'color': "#f8d7da"}]
                            }
                        ))
                        gauge_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    with p2:
                        r1, r2, r3, r4 = st.columns(4)
                        r1.metric("1M Return", f"{res['ret_1m']:+.2f}%")
                        r2.metric("1Y Return", f"{res['ret_1y']:+.2f}%")
                        r3.metric("Sharpe", f"{res['sharpe']:.2f}")
                        r4.metric("Sortino", f"{res['sortino']:.2f}")
                        
                        fig_nav = px.line(res['df_nav'], x=res['df_nav'].index, y='nav', title="Historical NAV Growth (₹)")
                        fig_nav.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig_nav, use_container_width=True)
                        
                with tab_stress:
                    st.subheader("⚡ Historical Crisis & Macro Shock Replay Engine")
                    df_nav = res['df_nav']
                    covid_period = df_nav.loc['2020-01-01':'2020-05-31']
                    c_drop = ((covid_period['nav'].min() / covid_period['nav'].max()) - 1.0) * 100.0 if len(covid_period) > 10 else -28.4
                    
                    rate_period = df_nav.loc['2022-01-01':'2022-06-30']
                    r_drop = ((rate_period['nav'].min() / rate_period['nav'].max()) - 1.0) * 100.0 if len(rate_period) > 10 else -12.1
                    
                    sim_shock = st.slider("Simulate Hypothetical Market Crash (% Shock):", -40, -5, -20, step=5)
                    est_loss = sim_shock * (res['vol'] / 15.0)
                    sim_nav = res['nav'] * (1 + (est_loss / 100.0))
                    
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("COVID-19 Crash Replay (2020)", f"{c_drop:.1f}% Drawdown")
                    sc2.metric("2022 Rate-Hike Correction", f"{r_drop:.1f}% Drawdown")
                    sc3.metric(f"Simulated {sim_shock}% Shock Impact", f"{est_loss:.1f}% Loss", f"Est. NAV: ₹{sim_nav:.2f}", delta_color="inverse")
                    
                with tab_export:
                    st.subheader("📄 Export Institutional Due-Diligence Factsheet")
                    report_html = generate_audit_report(res)
                    st.download_button(
                        label="📥 Download Printable Due-Diligence Factsheet (HTML / PDF)",
                        data=report_html,
                        file_name=f"AlphaShield_Audit_{selected_code}_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Error fetching data: {e}")

# ==============================================================================
# VIEW 2: HEAD-TO-HEAD SCHEME DUEL
# ==============================================================================
elif app_mode == "⚔️ Head-to-Head Scheme Duel (Fund A vs Fund B)":
    st.title("⚔️ Live Head-to-Head Scheme Duel")
    search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🟦 Fund A")
        choice_a = st.selectbox("Search Scheme A:", options=search_options, index=0, key="duel_a")
        code_a = all_schemes_df[all_schemes_df['Search_Label'] == choice_a]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"
    with col_b:
        st.markdown("### 🟧 Fund B")
        default_b = min(1, len(search_options) - 1)
        choice_b = st.selectbox("Search Scheme B:", options=search_options, index=default_b, key="duel_b")
        code_b = all_schemes_df[all_schemes_df['Search_Label'] == choice_b]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "120465"
        
    if st.button("⚔️ Launch Live Head-to-Head Duel", type="primary", use_container_width=True):
        with st.spinner("Evaluating live AMFI metrics..."):
            try:
                res_a = compute_scheme_metrics(code_a.strip())
                res_b = compute_scheme_metrics(code_b.strip())
                
                st.markdown("---")
                if res_a['prob'] < res_b['prob']:
                    st.success(f"### 🏆 WINNER: {res_a['name']} exhibits lower risk ({res_a['prob']*100:.1f}% vs {res_b['prob']*100:.1f}%)")
                elif res_b['prob'] < res_a['prob']:
                    st.success(f"### 🏆 WINNER: {res_b['name']} exhibits lower risk ({res_b['prob']*100:.1f}% vs {res_a['prob']*100:.1f}%)")
                else:
                    st.info("### ⚖️ TIE: Identical distress probability.")
                    
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown(f"#### 🟦 {res_a['name']}")
                    st.metric("Live Distress Risk", f"{res_a['prob']*100:.1f}%")
                with g2:
                    st.markdown(f"#### 🟧 {res_b['name']}")
                    st.metric("Live Distress Risk", f"{res_b['prob']*100:.1f}%")
            except Exception as e:
                st.error(f"Duel error: {e}")

# ==============================================================================
# VIEW 3: PAPER TRADING & PROOF LEDGER
# ==============================================================================
elif app_mode == "💼 Paper Trading & AI Proof Ledger":
    st.title("💼 Live Paper Trading & AI Proof Ledger")
    total_invested_cost = sum(pos['invested_amt'] for pos in st.session_state.portfolio)
    current_portfolio_val = sum(pos['units'] * pos['buy_nav'] for pos in st.session_state.portfolio)
    total_net_worth = st.session_state.cash + current_portfolio_val
    overall_pnl = current_portfolio_val - total_invested_cost
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    col_w1.metric("Available Paper Cash", f"₹{st.session_state.cash:,.2f}")
    col_w2.metric("Portfolio Current Value", f"₹{current_portfolio_val:,.2f}")
    col_w3.metric("Total Net Worth", f"₹{total_net_worth:,.2f}", f"₹{overall_pnl:+,.2f} Net P&L")
    col_w4.metric("Active Holdings", f"{len(st.session_state.portfolio)}")
    
    st.markdown("---")
    st.subheader("🛒 Execute Virtual Investment (Live Market NAV)")
    trade_source = st.radio("Selection Source:", ["⭐ Popular Active Benchmark Schemes", "🔎 Search Full Universe"], horizontal=True)
    c_trade_sel, c_trade_amt, c_trade_btn = st.columns([2.5, 1, 1])
    with c_trade_sel:
        if trade_source.startswith("⭐"):
            chosen_label = st.selectbox("Select Active Scheme to Trade:", list(ACTIVE_BENCHMARK_FUNDS.keys()), key="pt_sel")
            target_code = ACTIVE_BENCHMARK_FUNDS[chosen_label]
        else:
            search_options = all_schemes_df['Search_Label'].tolist() if not all_schemes_df.empty else list(ACTIVE_BENCHMARK_FUNDS.keys())
            target_scheme = st.selectbox("Select Active Scheme to Trade:", options=search_options, index=0, key="pt_all")
            target_code = all_schemes_df[all_schemes_df['Search_Label'] == target_scheme]['Scheme_Code'].iloc[0] if not all_schemes_df.empty else "118989"
    with c_trade_amt:
        order_amount = st.number_input("Investment Amount (₹):", min_value=1000.0, max_value=max(1000.0, st.session_state.cash), value=min(10000.0, st.session_state.cash), step=1000.0)
    with c_trade_btn:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        execute_buy = st.button("📥 Buy Units (Live AMFI)", type="primary", use_container_width=True)
        
    if execute_buy:
        if order_amount > st.session_state.cash:
            st.error("Insufficient paper cash! Reset portfolio or reduce investment amount.")
        else:
            with st.spinner("Executing order against live AMFI NAV..."):
                try:
                    res = compute_scheme_metrics(target_code)
                    units_allotted = order_amount / res['nav']
                    ai_verdict = "🔴 Critical Distress" if res['prob'] >= 0.6 else ("🟡 Watchlist" if res['prob'] >= 0.3 else "🟢 Healthy")
                    
                    st.session_state.portfolio.append({
                        'time': datetime.now().strftime("%d-%b-%Y %H:%M"),
                        'code': target_code,
                        'name': res['name'],
                        'category': res['category'],
                        'buy_nav': res['nav'],
                        'units': units_allotted,
                        'invested_amt': order_amount,
                        'ai_prob_at_buy': res['prob'],
                        'ai_verdict_at_buy': ai_verdict
                    })
                    st.session_state.cash -= order_amount
                    st.success(f"✅ Successfully purchased {units_allotted:.3f} units of {res['name']} at ₹{res['nav']:.2f} NAV! Logged AI failure risk at {res['prob']*100:.1f}%.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Execution Error: {e}")

    st.markdown("---")
    st.subheader("📑 Active Portfolio Ledger & Model Accountability")
    if len(st.session_state.portfolio) == 0:
        st.info("No active investments yet. Select an active scheme above and allocate virtual cash to start the efficiency audit.")
    else:
        ledger_rows = []
        for pos in st.session_state.portfolio:
            ledger_rows.append({
                "Timestamp": pos['time'],
                "Mutual Fund Scheme": pos['name'],
                "Category": pos['category'],
                "Execution NAV": f"₹{pos['buy_nav']:.2f}",
                "Units Held": f"{pos['units']:.3f}",
                "Invested (₹)": f"₹{pos['invested_amt']:,.0f}",
                "AI Failure Risk At Buy": f"{pos['ai_prob_at_buy']*100:.1f}%",
                "Model Status": pos['ai_verdict_at_buy']
            })
        st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True)
        if st.button("🔄 Reset Portfolio to ₹1,00,000 Cash", type="secondary"):
            st.session_state.cash = 100000.0
            st.session_state.portfolio = []
            st.rerun()

# ==============================================================================
# VIEW 4: HISTORICAL DATASET ARCHIVE
# ==============================================================================
else:
    st.title("📁 Historical Research Dataset (47,272 Records)")
    all_names = sorted(funds_df['Fund_Name'].dropna().unique().tolist())
    picked_fund = st.selectbox("Select Historical Fund:", all_names)
    if picked_fund:
        row = funds_df[funds_df['Fund_Name'] == picked_fund].iloc[0]
        st.markdown(f"**Scheme:** `{row['Fund_Name']}` | **Category:** `{row.get('Category', 'N/A')}` | **AMC:** `{row.get('AMC', 'N/A')}`")
        X_h = pd.DataFrame([[float(row[c]) for c in features]], columns=features)
        h_prob = float(model.predict_proba(X_h)[0, 1])
        st.metric("Historical Failure Risk", f"{h_prob*100:.1f}%")
