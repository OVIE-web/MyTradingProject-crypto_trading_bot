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

## 🔔 Notification Setup

### Telegram
1. Create a bot with [BotFather](https://t.me/BotFather) and get your bot token.
2. Start a chat with your bot and get your chat ID using:
   `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. Add these to your `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```
4. Use the Streamlit dashboard to send test messages.

### Email
1. Use an app password for Gmail or your SMTP provider.
2. Set the following in your `.env`:
   ```
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587
   EMAIL_USER=your_email@gmail.com
   EMAIL_PASS=your_app_password
   EMAIL_TO=your_email@gmail.com
   ```
3. Test by triggering a notification event in the app.

---

## 💻 Docker & Compose Usage

- The Dockerfile and docker-compose.yml are set up for production and local development.
- Model artifacts are stored in `src/models/` and shared via the `models_data` Docker volume.
- Logs are written to the `logs/` directory and mapped as a volume.
- The `.env` file is mapped read-only for security.
- To run the full stack:
  ```powershell
  docker-compose up --build
  ```
- To run the Streamlit dashboard only:
  ```powershell
  docker-compose run --rm tradingbot streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
  ```
- To train the model in Docker:
  ```powershell
  docker-compose run --rm tradingbot python main.py --train-only
  ```

---

## 🔒 Security & Best Practices

- The `.env` file is never committed and is mapped read-only in Docker Compose for extra safety.
- All secrets (API keys, DB passwords, JWT, etc.) are managed via environment variables.
- Use `.env.example` as a template and never share your real `.env`.
- If you need to persist models or logs, use Docker volumes as configured.
- For troubleshooting Docker build issues (e.g., network timeouts), try increasing build timeouts or retrying the build. See the Dockerfile for the `UV_HTTP_TIMEOUT` setting.

---

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
