# 🧠 Crypto Trading Bot 1

[![CI Status](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1/actions/workflows/tests.yml/badge.svg)](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1/actions/workflows/tests.yml)

A modular, type-safe, production-oriented crypto trading system built with **FastAPI**, **PostgreSQL**, **SQLAlchemy 2.0**, **XGBoost**, and **Plotly**, with a strong focus on correctness, security, and long-term maintainability.

---

## 🚀 Project Status (Current Phase)

**Status:** 🚧 Active Development  
**Current Focus:**  
✅ Type safety (mypy)  
✅ Configuration hardening  
✅ Authentication & JWT security  
✅ Backtesting correctness  
🟡 API stabilization  
🔜 Live trading hardening

This repository is currently in a **production-hardening phase**, prioritizing correctness and safety over feature expansion.

---

## 🧩 Core Capabilities (Implemented)

### 🔐 Authentication & Security

- JWT-based authentication
- Centralized, validated configuration via `settings.py`
- Environment-based safety checks (dev vs non-dev)
- No default or weak credentials in production environments

### 🗄️ Database Layer

- SQLAlchemy **2.0-style ORM**
- PostgreSQL support
- Typed session management
- Trade persistence model (`Trade`)

### 🧠 Machine Learning

- Feature-based signal generation (RSI, SMA, Bollinger Bands, ATR, momentum)
- XGBoost classifier integration
- Deterministic model loading & prediction paths
- Model lifecycle handled in `model_manager.py`

### 📊 Backtesting & Analytics

- Deterministic backtesting engine (`backtester.py`)
- Fee-aware trade execution logic
- Portfolio value tracking
- Win-rate, returns, and trade-level metrics
- Interactive Plotly visualizations (`visualizer.py`)

### 🌐 API Layer

- FastAPI application (`main_api.py`)
- Token issuance endpoint (`/token`)
- Trade inspection endpoints (`/trades`)
- Fully typed request/response models

---

## 🧱 Project Structure

crypto_trading_bot-1/
├── src/
│ ├── main_api.py # FastAPI entrypoint
│ ├── auth.py # JWT auth & password hashing
│ ├── settings.py # Strict, validated runtime settings
│ ├── config.py # Legacy + ML/trading config (being phased out)
│ ├── db.py # SQLAlchemy 2.0 ORM + session handling
│ ├── model_manager.py # ML model lifecycle
│ ├── backtester.py # Backtesting engine
│ ├── visualizer.py # Plotly analytics
│ ├── binance_manager.py # Binance Testnet integration
│ ├── notifier.py # Email / Telegram alerts
│ ├── bot_runner.py # Trading bot runtime
│ ├── streamlit_app.py # Analytics UI (optional)
│ └── init.py
│
├── tests/ # Unit & integration tests
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── .github/workflows/tests.yml

---

---

## ⚙️ Local Development Setup

### Prerequisites

- **Python ≥ 3.12**
- **uv**
- **Docker**
- **PostgreSQL**

### Setup

```bash
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1.git
cd MyTradingProject-crypto_trading_bot-1

uv venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

uv sync --extra dev


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

DATABASE_URL=postgresql://user:pass@localhost:5432/tradingbot
JWT_SECRET_KEY=<secure-random-32+ chars>
ADMIN_USERNAME=<admin_user>
ADMIN_PASSWORD=<strong_password>


# 5️⃣ Run tests
pytest -v --disable-warnings

## 🧠 Running the Application

# FastAPI backend
uvicorn src.main_api:app --reload

# Backtesting/Research
python -m src.backtester

# Telegram Bot
python -m src.bot_runner

# Streamlit Dashboard
streamlit run src/streamlit_app.py
```

## 🧪 Quality Gates

mypy for static typing

pytest for unit/integration tests

SQLAlchemy 2.0 typing

Logging-first (no prints)

Fail-fast configuration validation

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

No secrets committed

Strong runtime validation

JWT secrets enforced

Testnet-only trading by default

Explicit environment separation

## 🛠️ Future Enhancements

✅ Full mypy pass across codebase

🔜 JWT token hardening for /users/me

🔜 Binance live-trading safeguards

🔜 Async task queues (Redis)

🔜 Monitoring & metrics

🔜 Model versioning & ensembles

## 👤 Author: Ovie

Data Scientist & Machine Learning Engineer
Focused on correctness-first, production-grade systems.

## 📄License

MIT
Status: 🚧 Active Development
Last updated: January 2026
