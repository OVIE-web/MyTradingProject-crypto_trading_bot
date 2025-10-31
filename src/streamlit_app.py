import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add the repository root to Python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

# Initialize configuration from Streamlit secrets
if 'trading' not in st.secrets:
    st.error("Missing trading configuration in secrets!")
    st.stop()

TRADE_SYMBOL = st.secrets.trading.symbol
TRADE_INTERVAL = st.secrets.trading.interval
TRADE_QUANTITY = float(st.secrets.trading.quantity)
FEE_PCT = float(st.secrets.trading.fee_pct)

# Feature columns for the model
FEATURE_COLUMNS = ['rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct_b', 
                  'sma_20', 'sma_50', 'ma_cross', 'price_momentum']

# Try importing optional dependencies
try:
    from src.model_manager import load_trained_model, make_predictions
    from src.db import SessionLocal, Trade
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

# Page config
st.set_page_config(
    page_title="CryptoSignal Trading Assistant",
    page_icon="📈",
    layout="wide"
)

# Verify Telegram web app authentication if available
if 'telegram' in st.secrets:
    query_params = st.experimental_get_query_params()
    if 'hash' in query_params:
        # TODO: Implement Telegram WebApp authentication verification
        # https://core.telegram.org/bots/webapps#validating-data-received-via-the-web-app
        pass

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-title'>CryptoSignal Trading Assistant</h1>", unsafe_allow_html=True)

# Load model if dependencies are available
model = None
if HAS_DEPENDENCIES:
    @st.cache_resource
    def get_model():
        try:
            return load_trained_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    model = get_model()

# Layout with columns
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Market Analysis")
    with st.expander("Technical Indicators", expanded=True):
        user_input = {}
        for col in FEATURE_COLUMNS:
            user_input[col] = st.number_input(
                col, 
                value=0.0,
                help=f"Enter value for {col}"
            )
        
        if st.button("Generate Signal", use_container_width=True):
            X = pd.DataFrame([user_input])[FEATURE_COLUMNS]
            preds, probs = make_predictions(model, X)
            
            signal_map = {-1: "SELL 🔴", 0: "HOLD ⚪", 1: "BUY 🟢"}
            signal = signal_map.get(int(preds[0]), "HOLD ⚪")
            
            st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
                    <h3 style='text-align: center; margin: 0;'>Signal: {signal}</h3>
                    <p style='text-align: center; margin: 0.5rem 0;'>Confidence: {float(probs[0]):.2f}</p>
                </div>
            """, unsafe_allow_html=True)

with col1:
    st.markdown("### Portfolio Overview")
    
    # Get recent trades from database if available
    recent_trades = []
    if HAS_DEPENDENCIES and 'database' in st.secrets:
        try:
            # Create database session using secrets
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            engine = create_engine(st.secrets.database.url)
            SessionMaker = sessionmaker(bind=engine)
            db = SessionMaker()
            
            recent_trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(10).all()
            db.close()
            
            # Format SQLAlchemy objects
            trades_data = [{
                'Time': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Action': trade.side,
                'Price': f"${trade.price:.2f}",
                'Quantity': f"{trade.quantity:.4f} {TRADE_SYMBOL[:3]}"
            } for trade in recent_trades]
        except Exception as e:
            st.warning("Could not connect to database. Showing demo data.")
            # Create demo trades directly as dictionaries
            trades_data = [{
                'Time': (datetime.now() - timedelta(minutes=i*15)).strftime('%Y-%m-%d %H:%M'),
                'Action': 'BUY' if i % 2 == 0 else 'SELL',
                'Price': f"${40000 + (i * 100):.2f}",
                'Quantity': f"0.1 {TRADE_SYMBOL[:3]}"
            } for i in range(5)]
    else:
        # Create demo data when dependencies aren't available
        trades_data = [{
            'Time': (datetime.now() - timedelta(minutes=i*15)).strftime('%Y-%m-%d %H:%M'),
            'Action': 'BUY' if i % 2 == 0 else 'SELL',
            'Price': f"${40000 + (i * 100):.2f}",
            'Quantity': f"0.1 {TRADE_SYMBOL[:3]}"
        } for i in range(5)]
    
    if trades_data:
        
        st.dataframe(
            pd.DataFrame(trades_data),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No recent trades found.")

    # Placeholder for charts
    st.markdown("### Performance Chart")
    
    # Sample performance data (replace with real data)
    dates = pd.date_range(start='2025-01-01', end='2025-10-31', freq='D')
    performance = pd.DataFrame({
        'date': dates,
        'value': [1000 * (1 + i/100) for i in range(len(dates))]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance['date'],
        y=performance['value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1E88E5')
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Portfolio Value ($)',
        xaxis_title='Date'
    )
    
    st.plotly_chart(fig, use_container_width=True)