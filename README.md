# Cryptocurrency Trading Bot

A modular, production-ready cryptocurrency trading system using machine learning (XGBoost), technical indicators, and robust infrastructure for both backtesting and live trading. The project features a full pipeline from data collection and feature engineering to model training, backtesting, live trading, and interactive dashboards. It is designed for secure, scalable deployment using Docker and best practices.

## ✨ Features

*   **Data Collection & Preprocessing:** Handles historical and live OHLCV data loading and cleaning.
*   **Technical Analysis:** Calculates indicators such as:
    *   Relative Strength Index (RSI)
    *   Bollinger Bands (BB_upper, BB_lower, BB_mid, BB_pct_b)
    *   Simple Moving Averages (SMA_20, SMA_50)
    *   Moving Average Crossover Signal (MA_cross)
    *   Price Momentum
    *   Average True Range (ATR)
*   **Feature Engineering:** Generates trading signals and prepares features for the ML model.
*   **Model Training & Optimization:**
    *   Uses **XGBoost Classifier** for buy/sell/hold predictions
    *   Handles class imbalance with SMOTE
    *   Hyperparameter optimization via `RandomizedSearchCV`
*   **Backtesting Framework:** Simulates trades on historical data, calculates key metrics (Total Return, Win Rate, etc.)
*   **Interactive Visualization:**
    *   **Plotly**: Generates detailed, interactive charts (candlesticks, signals, indicators, portfolio value)
    *   **Streamlit**: Serves dashboards and analytics web apps, embedding Plotly charts for user interaction
*   **API Service:**
    *   **FastAPI**: REST API for model inference, health checks, and management
    *   Input validation, robust error handling, and secure authentication (API key & OAuth2/JWT)
*   **Experiment Tracking:**
    *   **MLflow**: Track model training, parameters, and results
*   **Database Integration:**
    *   **Postgres** with SQLAlchemy ORM for trade and user data
*   **Notifications:**
    *   Email and Telegram alerts for critical events
*   **Live Trading (Binance Testnet):**
    *   Real-time trading simulation with Binance Testnet API
*   **Dockerized & Orchestrated:**
    *   Dockerfile and docker-compose for seamless deployment of all services (bot, API, Streamlit, MLflow, Postgres)
*   **Security & Production Readiness:**
    *   Environment variables for secrets, JWT authentication, error logging, and best practices for deployment

## 🚀 Getting Started

### Prerequisites

*   **Python:** 3.12.x
*   **uv:** Fast Python package installer (`pip install uv`)
*   **Docker Desktop:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Installation & Setup

1.  **Clone or Prepare Project Structure:**
    ```
    MyTradingProject/
    └── crypto_trading_bot/
        ├── src/
        │   ├── models/          # Model storage (Docker volume in prod)
        │   ├── __init__.py
        │   ├── main_api.py      # FastAPI endpoints
        │   ├── backtester.py    # Trading simulation
        │   ├── binance_manager.py  # Exchange API
        │   ├── config.py        # Settings & env vars
        │   ├── data_loader.py   # OHLCV data handling
        │   ├── db.py           # Database models
        │   ├── feature_engineer.py  # Technical analysis
        │   ├── model_manager.py  # ML pipeline
        │   ├── notifier.py     # Alerts (email/telegram)
        │   ├── streamlit_app.py  # Dashboard
        │   └── visualizer.py   # Trading charts
        ├── tests/
        │   ├── __init__.py
        │   ├── conftest.py     # Pytest fixtures
        │   ├── test_backtester.py
        │   ├── test_binance_manager.py
        │   ├── test_data_loader.py
        │   ├── test_db.py
        │   ├── test_feature_engineer.py
        │   ├── test_model_manager.py
        │   ├── test_notification.py
        │   ├── test_notifier.py
        │   └── test_run_modes.py  # CLI & training tests
        ├── data/
        │   └── test_df_features.csv  # Historical data
        ├── main.py             # Entry point & CLI
        ├── requirements.txt    # Pinned dependencies
        ├── pyproject.toml     # Project & deps
        ├── pytest.ini         # Test settings
        ├── pg_hba.conf       # Postgres auth
        ├── .python-version   # Python 3.12.x
        ├── .env.example      # Environment template
        ├── .env             # Production vars
        ├── .env.local       # Development overrides
        ├── Dockerfile       # Container build
        ├── docker-compose.yml  # Service configs
        └── .dockerignore       # Build exclusions
    ```

2.  **Navigate to Project Directory:**
    ```powershell
    cd C:\Users\oviem\OneDrive\Desktop\Projects\MyTradingProject\crypto_trading_bot-1
    ```

3.  **Create and Activate Virtual Environment:**
    ```powershell
    uv venv
    .venv\Scripts\activate
    # For Cmd: .venv\Scripts\activate.bat
                .venv\Scripts\activate.ps1
    ```

4.  **Install Dependencies:**
    ```powershell
    uv pip install -e .
    # or
    uv pip install -r requirements.txt
    ```

5.  **Place Historical Data:**
    Ensure `test_df_features.csv` is in the `data/` directory of `crypto_trading_bot`.

6.  **Set up Environment Variables:**
    *   Copy `.env.example` to `.env` and fill in your secrets (Binance API keys, JWT secret, DB credentials, etc.)
    *   Example:
        ```
        BINANCE_API_KEY=YOUR_TESTNET_BINANCE_API_KEY
        BINANCE_API_SECRET=YOUR_TESTNET_BINANCE_API_SECRET
        JWT_SECRET_KEY=YOUR_RANDOM_SECRET
        ...
        ```
    *   **Never commit `.env` to public repos.**

## ⚙️ Configuration

All major parameters are in `src/config.py` and `.env`:
*   `TRADE_SYMBOL`, `TRADE_INTERVAL`, `TRADE_QUANTITY`
*   `CONFIDENCE_THRESHOLD` (model predictions)
*   Technical indicator windows (`RSI_WINDOW`, `BB_WINDOW`, `SMAs` etc.)
*   `MODE` (`live` or `backtest`)
*   `BINANCE_TESTNET` (True/False)
*   Database, email, Telegram, and JWT settings

## 💻 Usage

### Backtesting Mode (Local)

1.  Set `MODE=backtest` in `.env`.
2.  Activate your virtual environment.
3.  Run:
    ```powershell
    python main.py
    ```
    *   A Plotly chart will open in your browser (or via Streamlit dashboard if enabled).

### Live Trading Mode (Binance Testnet via Docker Compose)

**⚠️ EXTREME CAUTION: Live trading uses real API keys and can place orders. Always use the Testnet for safety.**

1.  Set `MODE=live` in `.env`.
2.  Build and start all services:
    ```powershell
    docker-compose up --build
    ```
    *   This launches the trading bot, FastAPI server (port 8000), Streamlit dashboard (port 8501), MLflow UI (port 5000), and Postgres DB (port 5432).
    *   Logs are printed to the terminal. Plotly charts are available via Streamlit.

### API & Dashboard

*   **FastAPI:** http://localhost:8000/docs (interactive API docs)
*   **Streamlit Dashboard:** http://localhost:8501
*   **MLflow Tracking UI:** http://localhost:5000

## 📁 Project Structure

Key files and folders:
*   `src/` — All Python modules
*   `src/main_api.py` — FastAPI app
*   `src/streamlit_app.py` — Streamlit dashboard
*   `src/model_manager.py` — ML model loading/inference
*   `src/db.py` — Database models and ORM
*   `src/feature_engineer.py` — Feature engineering
*   `src/binance_manager.py` — Binance API integration
*   `src/visualizer.py` — Plotly chart generation
*   `src/config.py` — Configuration
*   `main.py` — Entry point, trading logic
*   `data/` — Data files (e.g., test_df_features.csv)
*   `requirements.txt`, `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env`, `.dockerignore`

## 🔒 Security & Best Practices

*   All secrets and credentials are managed via `.env`.
*   API endpoints are protected with API key and OAuth2/JWT authentication.
*   Error handling, logging, and notifications are integrated.
*   Docker Compose orchestrates all services for production deployment.
*   Use strong, unique secrets for JWT and database credentials.
*   Regularly audit and update dependencies in `pyproject.toml` and `requirements.txt`.
*   Models are persisted in a Docker volume (`models_data`) and shared across services.

### Data & Model Persistence
*   **Models:** Saved to `/app/src/models` inside containers, shared via Docker volume
*   **Database:** PostgreSQL data in `postgres_data` volume
*   **Source Code:** Mounted from host (for development)
*   **Logs:** Written to host's `logs/` directory

You can train a model quickly using the CLI flag:
```powershell
# Local development (writes to src/models/)
python main.py --train-only

# In Docker (writes to models_data volume)
docker-compose run --rm tradingbot python main.py --train-only
```

## 📈 Visualization: Plotly & Streamlit

*   **Plotly** is used for generating interactive charts and analytics.
*   **Streamlit** serves these charts and dashboards to users via a web interface.

## 🛠️ Next Steps / TODO

-   Expand unit and integration tests in the existing `tests/` directory
-   Set up CI/CD for automated testing and deployment
-   Add advanced monitoring/alerting (Sentry, Prometheus, etc.)
-   Harden API security (user management, password hashing, HTTPS)
-   Expand Streamlit dashboards for analytics
-   Add or improve documentation
-   Keep Docker Compose and data paths in sync with any folder changes
---

For questions or contributions, please open an issue or pull request.
