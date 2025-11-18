# 🧠 Crypto Trading Bot 1

[![CI Status](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1/actions/workflows/tests.yml/badge.svg)](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1/actions/workflows/tests.yml)

A modular, production-ready crypto trading system built with FastAPI, PostgreSQL, XGBoost, and Telegram + Email notifications — powered by uv, Docker, and GitHub Actions CI.

---

## 🚀 Overview

This project provides an end-to-end automated trading pipeline:

- **Machine-learning signal generation**
- **Backtesting and visualization**
- **Live trading on Binance Testnet**
- **Notifications and dashboards**
- **Continuous Integration / Delivery**

Designed for scalability, reproducibility, and test-driven development.

---

## ✨ Key Features

---

### 🧠 Machine Learning & Strategy

XGBoost classifier for buy/sell/hold signals
RandomizedSearchCV + SMOTE for optimization and balance
Feature engineering with RSI, SMA, Bollinger Bands, ATR, Momentum

### 📊 Backtesting & Visualization

- Full performance metrics: CAGR, Sharpe Ratio, Max Drawdown  
- Plotly interactive charts and Streamlit analytics UI  
- Backtesting logic in `backtester.py`

### 🔗 API & Automation

- REST endpoints via **FastAPI** (`/predict`, `/train`, `/health`)
- **MLflow** experiment tracking  
- **SQLAlchemy + PostgreSQL** model registry  

### 🔔 Notification System

- Modular **Telegram Bot** + **SMTP Email** alerts  
- Centralized `notifier.py` with async retry logic  

### 🧩 Infrastructure & CI

- **Docker Compose** orchestrates FastAPI, Postgres, Streamlit, MLflow  
- **GitHub Actions CI** for linting, testing, and builds  
- **uv** manages reproducible environments  

---

## 🧱 Project Structure

crypto_trading_bot
  crypto_trading_bot-1/
├── src/
│   ├── main_api.py          # FastAPI entrypoint
│   ├── bot_runner.py        # Telegram bot
│   ├── notifier.py          # Alert dispatch (email/telegram)
│   ├── notification.py      # Notification utilities
│   ├── binance_manager.py   # Binance Testnet integration
│   ├── feature_engineer.py  # Technical indicators
│   ├── backtester.py        # Backtesting logic
│   ├── db.py                # SQLAlchemy ORM models
│   ├── config.py            # Global settings
│   ├── model_manager.py     # Model load/train/save
│   ├── visualizer.py        # Plotly charts
│   ├── streamlit_app.py     # Streamlit dashboard
│   └── -**init.py**      # src package initializer
│
├── tests/                   # Unit & integration tests
├── data/                    # Sample CSV datasets
├── requirements.txt
├── requirements.dev.txt
├── pyproject.toml
├── setup.py
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .python-version
├── pg_hba.conf
├── LICENSE
├── README.md
├── RulesFORAI.md
├── BeautyOfTheStruggle.md
├── crypto_trading_bot.ipynb
├── main.py
└── .github/workflows/tests.yml

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python ≥ 3.12**
- **uv** (modern package manager)
- **Docker Desktop**
- **PostgreSQL** (local or via Docker)

### Steps

```bash
# 1️⃣ Clone
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1.git
cd MyTradingProject-crypto_trading_bot-1

# 2️⃣ Create & activate virtual env
uv venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3️⃣ Sync dependencies
uv pip compile --extra dev pyproject.toml -o requirements.dev.txt
uv sync --frozen --extra dev

# 4️⃣ Set environment variables
setx TELEGRAM_BOT_TOKEN "123456789:ABC..."
setx TELEGRAM_CHAT_ID "987654321"
setx SMTP_HOST "smtp.gmail.com"
setx SMTP_PORT "587"
setx SMTP_USER "you@gmail.com"
setx SMTP_PASS "app_password"

# 5️⃣ Run tests
pytest -v --disable-warnings

## 🧠 Running the Application

# FastAPI backend
uvicorn src.main_api:app --reload

# Telegram Bot
python -m src.bot_runner

# Streamlit Dashboard
streamlit run src/streamlit_app.py
```

## 🐳 Docker Deployment

docker-compose up --build

Run individual modules:

docker-compose run --rm tradingbot python main.py --train-only

docker-compose run --rm tradingbot streamlit run src/streamlit_app.py

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (.github/workflows/tests.yml) automates:

Dependency installation via uv

Linting (black, flake8, isort, mypy)

Testing (pytest + Postgres service)

Secure secret management via GitHub secrets

- name: Run tests
  run: pytest tests/ --maxfail=1 --disable-warnings -q

## 🔒 Security Best Practices

Secrets stored only in GitHub or .env (never committed)

Docker services run with limited permissions

Use Binance Testnet API keys for safety

CI masks sensitive data automatically

🛠️ Future Enhancements

🧩 LSTM / Transformer model integration

📈 Real-time WebSocket signal dashboard

🧠 Model versioning & ensemble strategies

🔔 Async notification queues (Redis/RabbitMQ)

📊 Monitoring via Prometheus + Grafana

🧪 Coverage reports (Codecov)

## Author: Ovie

License: MIT
Status: 🚧 Active Development
Last Updated: November 2025
