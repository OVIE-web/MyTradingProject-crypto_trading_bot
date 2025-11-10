# 🧠 Crypto Trading Bot 1

[![CI Status](https://github.com/OVIE-web/crypto_trading_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/OVIE-web/crypto_trading_bot/actions/workflows/ci.yml)

A modular, test-driven crypto trading bot with XGBoost predictions, PostgreSQL registry, Telegram/Email notifications, and CI automation powered by GitHub Actions + uv.

## 📖 Description

    A **modular**, **production-ready cryptocurrency trading system** powered by **Machine Learning (XGBoost)**, **technical indicators**, and **automated notifications** (Telegram + Email).  
    This project provides a **full trading pipeline** — from data collection and model training to backtesting, live trading, and dashboard visualization.  
    Designed for **security**, **scalability**, and **continuous integration** with **Docker** and **GitHub Actions CI**.

    ---

    ## ✨ Key Features

    ### 🧠 Machine Learning & Strategy
    - **XGBoost Classifier** for buy/sell/hold signal prediction  
    - **SMOTE** for class balancing  
    - **RandomizedSearchCV** for hyperparameter optimization  
    - **Feature Engineering** with:
    - RSI, Bollinger Bands, SMA, MA Crossovers  
    - Momentum, ATR, and volatility measures  

    ### 📊 Backtesting & Visualization
    - **Backtesting engine** with metrics (CAGR, Sharpe Ratio, Win Rate, Max Drawdown)
    - **Plotly-powered interactive charts**
    - **Streamlit dashboard** for analytics and strategy testing

    ### 🔗 API & Automation
    - **FastAPI** backend for REST endpoints:
    - `/predict` → model inference  
    - `/train` → retrain model  
    - `/health` → system status  
    - **MLflow** for experiment tracking  
    - **SQLAlchemy + PostgreSQL** for persistence  

    ### 🔔 Notification System (New)
    - **Telegram bot integration** for alerts and trade signals  
    - **Email notifications (SMTP)** for critical events  
    - Centralized **notification manager** with modular `notifier.py` and `notification.py`
    - Supports async dispatch and error handling with retry logic  

    ### 🧩 Infrastructure & Deployment
    - **Dockerized services** for reproducible environments  
    - **docker-compose.yml** orchestrates:
    - FastAPI
    - PostgreSQL
    - Streamlit dashboard
    - MLflow tracker  
    - **GitHub Actions CI** for automatic linting, testing, and build validation  
    - Environment variables managed securely via `.env` and `.env.local`

    ---

    ## 🧱 Project Structure

    crypto_trading_bot/
    ├── src/
    │ ├── main_api.py # FastAPI entrypoint
    │ ├── bot_runner.py # Telegram bot runner
    │ ├── notifier.py # Handles sending alerts (email, telegram)
    │ ├── notification.py # Core notification utilities & wrappers
    │ ├── binance_manager.py # Binance Testnet API integration
    │ ├── feature_engineer.py # Technical indicators
    │ ├── backtester.py # Strategy simulation
    │ ├── db.py # SQLAlchemy ORM models
    │ ├── config.py # Global settings & environment handling
    │ ├── model_manager.py # Model load/train/save
    │ ├── visualizer.py # Plotly chart generator
    │ ├── streamlit_app.py # Interactive dashboard
    │ └── init.py
    │
    ├── notifications/
    │ ├── init.py
    │ ├── test_notifier.py
    │ ├── test_notifications.py
    │
    ├── tests/
    │ ├── init.py
    │ ├── conftest.py
    │ ├── test_backtester.py
    │ ├── test_binance_manager.py
    │ ├── test_data_loader.py
    │ ├── test_db.py
    │ ├── test_feature_engineer.py
    │ ├── test_model_manager.py
    │ ├── test_run_modes.py
    │ └── notifications/ # Notification-specific test module
    │
    ├── data/
    │ └── test_df_features.csv
    │
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── pyproject.toml
    ├── docker-compose.yml
    ├── Dockerfile
    ├── pytest.ini
    ├── .env.example
    ├── .env
    └── README.md


    ---

    ## 🧩 Installation & Setup

    ### Prerequisites
    - **Python 3.12.x**
    - **uv** (modern package manager) → `pip install uv`
    - **Docker Desktop**
    - **PostgreSQL** (local or via Docker)

    ### Steps

    ```bash
    # 1️⃣ Clone the repository
    git clone https://github.com/OVIE-web/crypto_trading_bot.git
    cd crypto_trading_bot

    # 2️⃣ Create a virtual environment
    uv venv
    .venv\Scripts\activate  # Windows
    source .venv/bin/activate  # Mac/Linux

    # 3️⃣ Install dependencies
    uv pip install -r requirements.txt
    uv pip install -r requirements-dev.txt

    # 4️⃣ Set environment variables (PowerShell example)
    setx TELEGRAM_BOT_TOKEN "123456789:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
    setx TELEGRAM_CHAT_ID "987654321"
    setx SMTP_HOST "smtp.gmail.com"
    setx SMTP_PORT "587"
    setx SMTP_USER "your_email@gmail.com"
    setx SMTP_PASS "app_password"

    # 5️⃣ Run tests
    pytest tests/ -v --disable-warnings


    🧠 Running the Application
    🧩 Local (FastAPI)
    uvicorn src.main_api:app --reload

    🤖 Telegram Bot
    python -m src.bot_runner

    📊 Streamlit Dashboard
    streamlit run src/streamlit_app.py

    🧰 Docker Deployment

    To start all services (API, Postgres, Streamlit, MLflow):

    docker-compose up --build


    To train only:

    docker-compose run --rm tradingbot python main.py --train-only


    To run Streamlit dashboard:

    docker-compose run --rm tradingbot streamlit run src/streamlit_app.py

    ⚙️ CI/CD Pipeline (GitHub Actions)

    The CI workflow (.github/workflows/ci.yml) automates:

    Dependency installation

    Test execution

    DB health check (Postgres service)

    Notification environment mocks

    Example Job
    - name: Run tests
    run: pytest tests/ --maxfail=1 --disable-warnings -q


    ✅ Tests run automatically on each push or pull_request to main.

    🔔 Notifications Configuration
    Telegram
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id

    Email
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=your_email@gmail.com
    SMTP_PASS=your_app_password


    Test manually:

    from src.notifier import TelegramNotifier
    TelegramNotifier().send_message("🚀 Test message successful!")

    🧪 Testing
    Run all tests
    pytest -v --disable-warnings

    Run only notifier tests
    pytest tests/notifications -v


    CI logs are streamed in GitHub Actions and visible in the Test Summary.

    🔒 Security Best Practices

    Secrets are never hardcoded — stored in .env or system variables

    .env is ignored by Git

    Use app passwords for email and Testnet keys for Binance

    Docker environment mappings are read-only

    CI masks sensitive data automatically

    🛠️ Future Enhancements

    🧩 Integrate deep learning models (LSTM/Transformers)

    📈 Real-time signal dashboard with WebSocket streaming

    🧠 Model versioning and ensemble strategies

    🔔 Async notification queues (Redis/RabbitMQ)

    📊 Prometheus + Grafana monitoring

    🧪 CI test coverage via Codecov

    Author: Ovie
    License: MIT
    Status: 🚧 Active Development
    Last: Updated November 2025