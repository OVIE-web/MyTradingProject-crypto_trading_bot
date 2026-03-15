# 🤖 Crypto Trading Bot

[![Security](https://img.shields.io/badge/security-scanning-green)](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/security/code-scanning)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi)
![Coverage](https://img.shields.io/badge/coverage-65%25-yellow)
![Type Checking](https://img.shields.io/badge/mypy-checked-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A **modular, production-ready cryptocurrency trading bot** with a FastAPI backend, machine learning-powered trade predictions, and real-time portfolio management.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

**Crypto Trading Bot** is an intelligent trading automation system designed for cryptocurrency markets. It combines real-time market data analysis with machine learning models to identify profitable trading opportunities and execute trades automatically.

### Key Capabilities

- **Automated Trading**: Real-time trade execution on Binance
- **ML-Powered Predictions**: XGBoost models for entry/exit signals
- **Technical Analysis**: RSI, Long/Short Moving Averages, Bollinger Bands, and custom indicators
- **Portfolio Management**: Real-time balance tracking and risk management
- **REST API**: FastAPI backend for programmatic access
- **Web Dashboard**: Streamlit frontend for visualization and monitoring
- **Backtesting Engine**: Validate strategies before live trading
- **Notification System**: Telegram and email alerts
- **Production Ready**: Docker containerization, CI/CD, security scanning

---

## ✨ Features

### Core Trading Features

- **Intelligent Signal Generation**
  - ML-based entry/exit signal predictions
  - Technical indicator analysis (RSI, MACD, Bollinger Bands)
  - Custom feature engineering pipeline
  - Real-time backtesting and validation

- **Trade Execution**
  - Market order execution on Binance
  - Automated position sizing based on risk
  - Stop-loss and take-profit management
  - Slippage and fee optimization

- **Portfolio Management**
  - Real-time balance and position tracking
  - Profit/loss calculations
  - Risk metrics and exposure monitoring
  - Historical trade analytics

### API & Integration

- **RESTful API** (FastAPI)
  - Trade history endpoints
  - Portfolio status queries
  - Real-time performance metrics
  - Strategy configuration management

- **Web Dashboard** (Streamlit)
  - Live trading performance visualization
  - Portfolio overview and metrics
  - Technical indicator charts
  - Trade history and analysis

### DevOps & Quality

- **Containerization**: Docker and Docker Compose
- **Continuous Integration**: GitHub Actions with automated testing
- **Security Scanning**: CodeQL for vulnerability detection
- **Code Quality**: Ruff linting, MyPy type checking
- **Testing**: Comprehensive pytest suite (59+ tests, 45%+ coverage)
- **Database**: PostgreSQL for trade history and models

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Crypto Trading Bot                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Client Layer (Web & API)                    │   │
│  │  ├─ Streamlit Dashboard (Port 8501)                 │   │
│  │  ├─ FastAPI REST API (Port 8000)                    │   │
│  │  └─ Health Checks & Monitoring                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Core Trading Engine (Async)                   │   │
│  │  ├─ Market Data Fetcher (Binance API)              │   │
│  │  ├─ Technical Indicator Calculator                 │   │
│  │  ├─ ML Model Inference (XGBoost)                   │   │
│  │  ├─ Signal Generator & Validator                   │   │
│  │  ├─ Trade Executor (Order Management)              │   │
│  │  └─ Risk Manager & Position Sizer                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      Data & Persistence Layer                        │   │
│  │  ├─ PostgreSQL (Trade History, Models)             │   │
│  │  ├─ Feature Store (Preprocessed Data)              │   │
│  │  ├─ Model Registry (XGBoost Artifacts)             │   │
│  │  └─ MLflow (Experiment Tracking)                   │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │      External Services & Integrations               │   │
│  │  ├─ Binance API (Market Data & Orders)             │   │
│  │  ├─ Telegram Bot (Notifications)                   │   │
│  │  ├─ Email Service (Alerts)                         │   │
│  │  └─ Optional: Google Cloud Platform                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Binance API credentials (free account)
- PostgreSQL 13+ (or use Docker)

### Local Development

```bash
# Clone repository
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot.git
cd MyTradingProject-crypto_trading_bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install uv
uv pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your Binance API keys and settings

# Start services
docker-compose up -d

# Run tests
uv run python -m pytest -v

# Start bot
docker-compose run --rm bot
```

### Docker Quick Start

```bash
# Build and start all services
docker-compose up -d --build

# Monitor logs
docker-compose logs -f bot

# Access services
# Streamlit: http://localhost:8501
# API Docs: http://localhost:8000/docs
# PgAdmin: http://localhost:5050
```

---

## 💻 Technology Stack

### Backend & Core

| Component | Technology | Purpose |
| --- | --- | --- |
| Framework | FastAPI | REST API and async task handling |
| Language | Python 3.12 | Core trading logic |
| Async Runtime | AsyncIO | Non-blocking I/O operations |
| Task Scheduling | APScheduler | Recurring trading iterations |

### Machine Learning

| Component | Technology | Purpose |
| --- | --- | --- |
| ML Framework | XGBoost | Trade signal prediction |
| Data Processing | Pandas, NumPy | Feature engineering |
| Technical Analysis | TA-Lib | Indicator calculations |
| Experiment Tracking | MLflow | Model versioning and monitoring |
| Data Science | Scikit-learn | Preprocessing and metrics |

### Data & Database

| Component | Technology | Purpose |
| --- | --- | --- |
| Primary DB | PostgreSQL 16 | Trade history and models |
| ORM | SQLAlchemy 2.0 | Database abstraction |
| Migrations | Alembic | Schema versioning |

### Frontend & Monitoring

| Component | Technology | Purpose |
| --- | --- | --- |
| Dashboard | Streamlit | Real-time visualization |
| Charting | Plotly | Interactive performance charts |
| Notifications | Telegram Bot API | Real-time alerts |

### DevOps & Infrastructure

| Component | Technology | Purpose |
| --- | --- | --- |
| Containerization | Docker & Docker Compose | Local development and deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Code Quality | Ruff, MyPy | Linting and type checking |
| Testing | Pytest | Unit and integration tests |
| Security | CodeQL | Vulnerability scanning |

### External APIs

- **Binance API**: Market data, order execution, account management
- **Telegram Bot API**: Real-time notifications
- **SMTP**: Email alerts

---

## 📁 Project Structure

```
crypto-trading-bot/
├── src/
│   ├── bot_runner.py              # Main trading loop orchestrator
│   ├── backtester.py              # Strategy backtesting engine
│   ├── binance_manager.py         # Binance API integration
│   ├── data_loader.py             # Data fetching and preprocessing
│   ├── feature_engineer.py        # Technical indicator calculations
│   ├── model_manager.py           # ML model training and inference
│   ├── notification.py            # Email & Telegram notifications
│   ├── notifier.py                # Notification orchestration
│   ├── db.py                      # Database models and setup
│   ├── settings.py                # Environment configuration
│   ├── main_api.py                # FastAPI application
│   ├── streamlit_app.py           # Dashboard frontend
│   └── routers/
│       ├── predict.py             # Prediction endpoints
│       └── trades.py              # Trade history endpoints
│
├── tests/
│   ├── test_settings.py           # Configuration validation
│   ├── test_backtester.py         # Backtesting logic
│   ├── test_binance_manager.py    # API integration
│   ├── test_feature_engineer.py   # Indicator calculations
│   ├── test_model_manager.py      # ML pipeline
│   ├── test_data_loader.py        # Data processing
│   └── notifications/
│       └── test_notification.py   # Alert system
│
├── .github/
│   ├── workflows/
│   │   └── test.yml               # CI/CD pipeline
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml
│       └── feature_request.yml
│
├── scripts/
│   ├── init_database.py           # Database initialization
│   └── wait_for_postgres.py       # Connection polling
│
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml             # Local development setup
├── pyproject.toml                 # Project metadata & dependencies
├── pytest.ini                      # Test configuration
├── CODE_OF_CONDUCT.md             # Community guidelines
├── CONTRIBUTING.md                # Contribution guide
├── RulesFORAI.md                  # Standards for AI interaction
├── LICENSE                        # MIT License
└── README.md                       # This file
```

---

## 🛠️ Development

### Running Tests

```bash
# All tests with coverage
uv run python -m pytest -v --cov=src

# Specific test file
uv run python -m pytest tests/test_settings.py -v

# With HTML coverage report
uv run python -m pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint and check
uv run ruff check src/ tests/

# Type checking
uv run python -m mypy --config-file pyproject.toml src tests
uv run python -m mypy --config-file mypy.ini src tests

# All checks
uv run python -m pytest && uv run ruff format . && uv run python -m mypy --config-file pyproject.toml src tests && uv run python -m mypy --config-file mypy.ini src tests
```

### Test Coverage

- **Current**: 45.44% (59 tests passing, 1 skipped)
- **Target**: 70%+ minimum, 85%+ goal
- **Critical Paths**: 90%+ coverage

---

## 📊 Example Workflow

```python
# 1. Fetch market data
data = await binance_manager.get_latest_ohlcv("BTCUSDT", interval="1h")

# 2. Calculate technical indicators
features = feature_engineer.calculate_technical_indicators(data)

# 3. Generate ML prediction
signal = model_manager.predict(features)  # 0=Sell, 1=Hold, 2=Buy

# 4. Validate signal
if signal == 2 and validate_trade_conditions(data):
    # 5. Execute trade
    order = await binance_manager.place_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=calculate_position_size()
    )
    
    # 6. Send notification
    await notifier.send_telegram(f"🟢 BUY signal executed: {order}")
```

---

## 🔒 Security

- **Environment variables**: Sensitive data via `.env`
- **API keys**: Never committed to repository
- **CodeQL scanning**: Automated vulnerability detection
- **Type safety**: Full type hints with mypy checking
- **Input validation**: Pydantic models for all inputs
- **Database**: Encrypted connections and secure credentials

---

## 📈 Performance Metrics

- **Test Coverage**: 45.44% (improving with each contribution)
- **API Response Time**: < 200ms average
- **Trade Execution**: Real-time (≤500ms order placement)
- **Data Processing**: 100+ instruments/minute
- **Model Inference**: < 50ms per prediction
- **Code Quality**: 100% mypy checked, ruff compliant

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup instructions
- Coding standards and conventions
- Testing requirements
- Pull request process
- Community guidelines

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Binance API**: Market data and order execution
- **XGBoost**: Machine learning predictions
- **FastAPI**: Modern web framework
- **Streamlit**: Rapid dashboard development
- **PostgreSQL**: Reliable data storage
- **Community**: All contributors and supporters

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/discussions)
- **Email**: [oviemsaniyo30@gmail.com](mailto:oviemsaniyo30@gmail.com)

---

## ⚠️ Disclaimer

**This software is provided for educational purposes only.** Cryptocurrency trading involves significant financial risk. Past performance does not guarantee future results. Always:

- Start with small amounts in test environments
- Use proper risk management strategies
- Never trade with funds you cannot afford to lose
- Thoroughly test strategies before live trading
- Keep API keys secure and rotated regularly

---

## 🗺️ Roadmap

- [ ] Multi-pair concurrent trading
- [ ] Advanced risk management (portfolio-level)
- [ ] Real-time model retraining
- [ ] Advanced charting and analytics
- [ ] Mobile app integration
- [ ] Backtesting optimization (parallel processing)
- [ ] More exchange integrations (Kraken, Coinbase)
- [ ] Community strategy sharing
- [ ] Cloud deployment guides (GCP, AWS, Azure)

---

**Made with ❤️ by the Crypto Trading Bot Team**

⭐ If you find this useful, please star the repository!
