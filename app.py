import streamlit as st
import pandas as pd
import yfinance as yf
import mplfinance as mpf
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Deep Quant AI", page_icon="🛰️", layout="wide")

# ==========================================
# 2. DESIGN SYSTEM — GLOBAL THEME
# ==========================================
def inject_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-void: #080B10;
        --bg-panel: #10151D;
        --bg-panel-hover: #161D27;
        --border-hair: #1F2733;
        --text-hi: #EAF0F9;
        --text-lo: #8C99AD;
        --bull: #2EE6A6;
        --bear: #FF5C72;
        --amber: #FFB84D;
        --signal-blue: #4DA8FF;
    }

    html, body, .stApp { background: var(--bg-void); font-family: 'Inter', sans-serif; }
    .stApp { color: var(--text-hi); }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; max-width: 1180px; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: var(--border-hair); border-radius: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }

    button:focus-visible, input:focus-visible { outline: 2px solid var(--signal-blue); outline-offset: 2px; }
    input[type="radio"] { accent-color: var(--signal-blue); }

    .qp-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.18em;
        text-transform: uppercase; color: var(--signal-blue); margin-bottom: 0.5rem; }
    .qp-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.4rem;
        color: var(--text-hi); margin: 0; line-height: 1.1; }
    .qp-subtitle { font-family: 'Inter', sans-serif; color: var(--text-lo); font-size: 1rem;
        margin-top: 0.6rem; max-width: 620px; line-height: 1.5; }
    .qp-scanline { height: 2px; width: 100%; border-radius: 2px; margin: 1.25rem 0 1.75rem;
        background: linear-gradient(90deg, transparent, var(--signal-blue), transparent);
        background-size: 200% 100%; animation: qp-scan 3.2s linear infinite; }
    @keyframes qp-scan { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

    .qp-brand { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; }
    .qp-brand-mark { font-size: 1.4rem; color: var(--signal-blue); }
    .qp-brand-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1rem;
        color: var(--text-hi); letter-spacing: 0.02em; }
    .qp-brand-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.12em;
        color: var(--text-lo); text-transform: uppercase; margin-top: 1px; }

    .qp-status-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.75rem 0 1.25rem; }
    .qp-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--text-lo); background: var(--bg-panel);
        border: 1px solid var(--border-hair); padding: 0.3rem 0.6rem; border-radius: 20px;
        display: inline-flex; align-items: center; }
    .qp-badge-bad { color: var(--bear); border-color: rgba(255,92,114,0.35); }
    .qp-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--bull); display: inline-block;
        margin-right: 6px; animation: qp-pulse 1.6s ease-in-out infinite; }
    @keyframes qp-pulse { 0%,100% { opacity: 1; box-shadow: 0 0 0 0 rgba(46,230,166,0.45); }
        50% { opacity: 0.65; box-shadow: 0 0 0 5px rgba(46,230,166,0); } }
    .qp-nav-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em;
        color: var(--text-lo); text-transform: uppercase; margin: 1.25rem 0 0.4rem; }
    .qp-sidebar-footer { font-family: 'Inter', sans-serif; font-size: 0.72rem; color: var(--text-lo);
        margin-top: 2rem; line-height: 1.5; border-top: 1px solid var(--border-hair); padding-top: 0.9rem; }

    .qp-card { background: var(--bg-panel); border: 1px solid var(--border-hair); border-radius: 14px;
        padding: 1.1rem 1.4rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
    .qp-alert-bad { border-left: 4px solid var(--bear); color: var(--text-hi); }
    .qp-alert-warn { border-left: 4px solid var(--amber); color: var(--text-hi); }
    .qp-card code { background: rgba(255,255,255,0.06); padding: 0.1rem 0.4rem; border-radius: 4px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }

    .qp-field-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-lo); margin-bottom: 0.35rem; height: 0.9rem; }
    .qp-section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em;
        text-transform: uppercase; color: var(--text-lo); margin: 1.6rem 0 0.8rem; padding-top: 1rem;
        border-top: 1px solid var(--border-hair); }
    .qp-card-header { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem;
        color: var(--text-hi); letter-spacing: 0.02em; margin-bottom: 0.6rem; }
    .qp-inline-success { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--bull);
        margin: 0.6rem 0 1.1rem; }
    .qp-inline-note { font-family: 'Inter', sans-serif; font-size: 0.78rem; color: var(--text-lo);
        margin: 0.4rem 0 1rem; line-height: 1.5; }

    .qp-verdict { animation: qp-fade-in 0.5s ease both; }
    @keyframes qp-fade-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .qp-verdict-glyph { font-size: 1.6rem; line-height: 1; margin-bottom: 0.3rem; }
    .qp-verdict-label { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5rem; }
    .qp-verdict-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em;
        color: var(--text-lo); text-transform: uppercase; margin-bottom: 0.9rem; }
    .qp-bar-track { width: 100%; height: 6px; background: rgba(255,255,255,0.07); border-radius: 6px;
        overflow: hidden; margin-bottom: 0.4rem; }
    .qp-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
    .qp-verdict-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: var(--text-lo);
        margin-bottom: 0.9rem; }
    .qp-verdict-msg { font-family: 'Inter', sans-serif; font-size: 0.88rem; color: var(--text-hi);
        line-height: 1.5; margin: 0; opacity: 0.9; }

    .qp-metrics-row { display: flex; gap: 1rem; flex-wrap: wrap; animation: qp-fade-in 0.5s ease both; }
    .qp-metric { flex: 1; min-width: 160px; background: var(--bg-panel); border: 1px solid var(--border-hair);
        border-radius: 14px; padding: 1.1rem 1.3rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
    .qp-metric-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.12em;
        color: var(--text-lo); text-transform: uppercase; margin-bottom: 0.5rem; }
    .qp-metric-value { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.7rem;
        color: var(--text-hi); }
    .qp-metric-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-top: 0.35rem; }
    .qp-metric-delta-muted { color: var(--text-lo); }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, var(--signal-blue), #2E7FE0);
        color: #051018; font-family: 'Space Grotesk', sans-serif; font-weight: 600;
        border: none; border-radius: 10px; padding: 0.6rem 1.2rem; letter-spacing: 0.01em;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px); box-shadow: 0 6px 18px rgba(77,168,255,0.35);
    }
    div[data-testid="stTextInput"] input {
        background: var(--bg-panel); border: 1px solid var(--border-hair); color: var(--text-hi);
        font-family: 'JetBrains Mono', monospace; border-radius: 8px; padding: 0.55rem 0.8rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--signal-blue); box-shadow: 0 0 0 3px rgba(77,168,255,0.15);
    }
    div[data-testid="stSlider"] label p { color: var(--text-lo) !important; font-size: 0.85rem; }
    div[data-testid="stSlider"] div[role="slider"] { background-color: var(--signal-blue) !important; }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--signal-blue) !important; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-panel) !important; border: 1px solid var(--border-hair) !important;
        border-radius: 14px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }

    div[data-testid="stAlert"] {
        background: var(--bg-panel); border: 1px solid var(--border-hair); border-radius: 12px;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-panel); border-right: 1px solid var(--border-hair);
    }

    img { border-radius: 8px; }
    div[data-testid="stImage"] figcaption {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.04em;
        color: var(--text-lo) !important; text-transform: uppercase;
    }

    @media (prefers-reduced-motion: reduce) {
        .qp-scanline, .qp-dot, .qp-verdict, .qp-metrics-row { animation: none !important; }
    }
    @media (max-width: 768px) {
        .qp-title { font-size: 1.7rem; }
        .qp-metrics-row { flex-direction: column; }
    }
    </style>
    """, unsafe_allow_html=True)

inject_theme()

# ==========================================
# 3. AI MODEL LOADING
#    Path is resolved relative to this script's own location, not the
#    current working directory, so it works regardless of where
#    `streamlit run` is invoked from.
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "financial_resnet.pth")


@st.cache_resource
def load_ai_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 3)
    )

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        return model, device, None
    except Exception as e:
        return None, None, str(e)


model, device, model_load_error = load_ai_model()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_live_chart_image(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="45d")

    if data.empty:
        return None, None

    data = data.dropna()
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    chunk = data.tail(30)

    mc = mpf.make_marketcolors(up='#2EE6A6', down='#FF5C72', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', facecolor='#10151D',
                            edgecolor='#10151D', figcolor='#10151D')

    buf = io.BytesIO()
    mpf.plot(chunk, type='candle', style=s, axisoff=True,
              savefig=dict(fname=buf, format='png', dpi=70, bbox_inches='tight', pad_inches=0))
    buf.seek(0)

    image = Image.open(buf).convert('RGB')
    return image, chunk


def run_time_machine(df, conf_thresh, stop_loss, take_profit, max_days):
    capital = 10000.0
    shares_held = 0.0
    buy_price = 0.0
    days_held = 0
    in_position = False
    trades, wins = 0, 0

    for index, row in df.iterrows():
        current_price = row['Close']
        ai_pred = row['AI_Prediction']
        ai_conf = row['AI_Confidence']

        if not in_position:
            if ai_pred == "bullish" and ai_conf >= conf_thresh:
                shares_held = capital / current_price
                buy_price = current_price
                capital = 0.0
                in_position = True
                days_held = 0
        else:
            days_held += 1
            current_pnl_pct = (current_price - buy_price) / buy_price
            sell_reason = None

            if current_pnl_pct <= stop_loss:
                sell_reason = "STOP LOSS"
            elif current_pnl_pct >= take_profit:
                sell_reason = "TAKE PROFIT"
            elif days_held >= max_days:
                sell_reason = "TIME OUT"

            if sell_reason:
                capital = shares_held * current_price
                shares_held = 0.0
                in_position = False
                trades += 1
                if current_price > buy_price:
                    wins += 1

    if in_position:
        capital = shares_held * df.iloc[-1]['Close']

    return capital, trades, wins


def render_hero(eyebrow, title, subtitle):
    st.markdown(f"""
    <div class="qp-eyebrow">{eyebrow}</div>
    <div class="qp-title">{title}</div>
    <div class="qp-subtitle">{subtitle}</div>
    <div class="qp-scanline"></div>
    """, unsafe_allow_html=True)


def render_verdict_card(prediction, conf_pct):
    config = {
        "BULLISH": {"color": "var(--bull)", "glyph": "▲",
                    "msg": "The model reads recent price structure as more consistent with its 'bullish' training examples than the other two classes."},
        "BEARISH": {"color": "var(--bear)", "glyph": "▼",
                    "msg": "The model reads recent price structure as more consistent with its 'bearish' training examples than the other two classes."},
        "NEUTRAL": {"color": "var(--amber)", "glyph": "■",
                    "msg": "The model reads this chart as range-bound, closer to its 'neutral' training examples than a clear directional pattern."},
    }[prediction]
    st.markdown(f"""
    <div class="qp-card qp-verdict" style="border-left: 4px solid {config['color']};">
        <div class="qp-verdict-glyph" style="color: {config['color']};">{config['glyph']}</div>
        <div class="qp-verdict-label" style="color: {config['color']};">{prediction}</div>
        <div class="qp-verdict-sub">AI Forecast</div>
        <div class="qp-bar-track"><div class="qp-bar-fill" style="width: {conf_pct:.1f}%; background: {config['color']};"></div></div>
        <div class="qp-verdict-conf">{conf_pct:.2f}% softmax confidence</div>
        <p class="qp-verdict-msg">{config['msg']}</p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="qp-brand">
        <div class="qp-brand-mark">◆</div>
        <div>
            <div class="qp-brand-title">DEEP QUANT</div>
            <div class="qp-brand-sub">AI Market Terminal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is not None:
        st.markdown(f"""
        <div class="qp-status-row">
            <span class="qp-badge"><span class="qp-dot"></span>Model online</span>
            <span class="qp-badge">{str(device).upper()}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="qp-status-row"><span class="qp-badge qp-badge-bad">Model offline</span></div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="qp-nav-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Live Oracle", "Interactive Backtester"], label_visibility="collapsed")

    st.markdown("""
    <div class="qp-sidebar-footer">
        Research tool for exploring model behavior on historical and live charts.
        Trained and backtested with a time-based train/test split to avoid
        evaluating the model on data it trained on. Outputs are probabilistic
        pattern reads, not financial advice, and have not been validated
        against live trading.
    </div>
    """, unsafe_allow_html=True)

if model is None:
    st.markdown(f"""
    <div class="qp-card qp-alert-bad">
        <strong>Model not found.</strong><br/>
        Looked for <code>{MODEL_PATH}</code> and failed.<br/>
        <span style="opacity:0.75; font-size:0.85em;">{model_load_error}</span><br/><br/>
        Run <code>model_training.py</code> first to produce <code>financial_resnet.pth</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==========================================
# PAGE 1: LIVE ORACLE
# ==========================================
if page == "Live Oracle":
    render_hero(
        "AI Market Analysis",
        "Live Market Oracle",
        "Pull today's price structure and run it through the trained ResNet50 vision model to read the market's current shape."
    )

    input_col, button_col = st.columns([3, 1])
    with input_col:
        st.markdown('<div class="qp-field-label">Ticker symbol</div>', unsafe_allow_html=True)
        ticker = st.text_input("Ticker", "AAPL", label_visibility="collapsed").upper()
    with button_col:
        st.markdown('<div class="qp-field-label">&nbsp;</div>', unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Chart", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"Pulling live market data and generating vision tensor for {ticker}..."):
            img, raw_data = get_live_chart_image(ticker)

        if img is None:
            st.markdown(f"""
            <div class="qp-card qp-alert-bad">Failed to fetch data for <strong>{ticker}</strong>. Check the symbol and try again.</div>
            """, unsafe_allow_html=True)
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(img).unsqueeze(0).to(device)

            categories = ['BEARISH', 'BULLISH', 'NEUTRAL']
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)

            prediction = categories[predicted_idx.item()]
            conf_pct = confidence.item() * 100

            st.markdown(f'<div class="qp-section-label">Results · {ticker}</div>', unsafe_allow_html=True)
            chart_col, verdict_col = st.columns([3, 2])
            with chart_col:
                with st.container(border=True):
                    st.image(img, caption="AI Vision Input · Last 30 Trading Days", use_container_width=True)
            with verdict_col:
                render_verdict_card(prediction, conf_pct)

# ==========================================
# PAGE 2: INTERACTIVE BACKTESTER
# ==========================================
elif page == "Interactive Backtester":
    render_hero(
        "Strategy Simulation",
        "Interactive Backtester",
        "Replay the model's predictions on its held-out test period against custom risk parameters."
    )

    st.markdown("""
    <div class="qp-inline-note">
        The data loaded here comes from <code>scanned_data/</code>, which
        <code>ai_scanner.py</code> restricts to each ticker's TEST period —
        the period <code>model_training.py</code> never trains on. Moving the
        sliders below still searches over the same fixed window shown here;
        for a genuine walk-forward check with a separate search/holdout split,
        see the console output of <code>backtester.py</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="qp-field-label">Ticker to load from scanned_data</div>', unsafe_allow_html=True)
    ticker = st.text_input("Backtest Ticker", "AAPL", label_visibility="collapsed").upper()
    filepath = os.path.join(BASE_DIR, "scanned_data", f"{ticker}_scanned.csv")

    if not os.path.exists(filepath):
        st.markdown(f"""
        <div class="qp-card qp-alert-warn">
            No scanned data found for <strong>{ticker}</strong>. Run <code>ai_scanner.py</code> on this ticker first.
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.read_csv(filepath)
        st.markdown(f"""
        <div class="qp-inline-success"><span class="qp-dot"></span>Loaded {len(df)} held-out test-period days for {ticker}</div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="qp-card-header">Adjust trading rules</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                conf_thresh = st.slider("Min AI Confidence (%)", 33, 99, 60)
                stop_loss = st.slider("Stop Loss (%)", -20.0, -1.0, -4.0) / 100.0
            with col2:
                take_profit = st.slider("Take Profit (%)", 1.0, 30.0, 8.0) / 100.0
                max_days = st.slider("Max Hold Time (Days)", 1, 60, 10)

            st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
            run_btn = st.button("Run Time Machine Simulation", type="primary", use_container_width=True)

        if run_btn:
            final_cash, trades, wins = run_time_machine(df, conf_thresh, stop_loss, take_profit, max_days)
            profit = final_cash - 10000.0
            win_rate = (wins / trades * 100) if trades > 0 else 0
            profit_color = "var(--bull)" if profit >= 0 else "var(--bear)"
            sign = "+" if profit >= 0 else ""

            st.markdown(f"""
            <div class="qp-section-label">Simulation results · starting capital $10,000</div>
            <div class="qp-metrics-row">
                <div class="qp-metric">
                    <div class="qp-metric-label">Final Balance</div>
                    <div class="qp-metric-value">${final_cash:,.2f}</div>
                    <div class="qp-metric-delta" style="color: {profit_color};">{sign}${profit:,.2f}</div>
                </div>
                <div class="qp-metric">
                    <div class="qp-metric-label">Total Trades</div>
                    <div class="qp-metric-value">{trades}</div>
                    <div class="qp-metric-delta qp-metric-delta-muted">executed</div>
                </div>
                <div class="qp-metric">
                    <div class="qp-metric-label">Win Rate</div>
                    <div class="qp-metric-value">{win_rate:.1f}%</div>
                    <div class="qp-metric-delta qp-metric-delta-muted">{wins}/{trades} wins</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if trades == 0:
                st.markdown("""
                <div class="qp-inline-note" style="margin-top: 1rem;">
                    No trades triggered with these parameters on this ticker's test window —
                    try lowering the confidence threshold.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="qp-inline-note" style="margin-top: 1rem;">
                    This single-window result is exploratory. For a check against overfitting,
                    run <code>backtester.py</code>, which fits parameters on one window and
                    reports performance on a separate holdout the search never saw.
                </div>
                """, unsafe_allow_html=True)
