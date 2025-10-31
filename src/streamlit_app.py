import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.model_manager import load_trained_model, make_predictions
from src.config import FEATURE_COLUMNS, TRADE_SYMBOL
from src.db import SessionLocal, Trade

# Page config
st.set_page_config(
    page_title="CryptoSignal Trading Assistant",
    page_icon="📈",
    layout="wide"
)

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

# Load model
@st.cache_resource
def get_model():
    return load_trained_model()

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
    
    # Get recent trades from database
    db = SessionLocal()
    recent_trades = db.query(Trade).order_by(Trade.timestamp.desc()).limit(10).all()
    db.close()
    
    if recent_trades:
        # Format trade data
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                'Time': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Action': trade.side,
                'Price': f"${trade.price:.2f}",
                'Quantity': f"{trade.quantity:.4f} {TRADE_SYMBOL[:3]}"
            })
        
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