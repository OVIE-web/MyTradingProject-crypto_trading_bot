# 🤖 Crypto Trading Bot

[![CI](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/actions)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![Type%20Checked](https://img.shields.io/badge/mypy-checked-blueviolet)
![License](https://img.shields.io/badge/license-MIT-green)

A **modular, production-ready cryptocurrency trading bot** with a FastAPI backend, machine learning-powered trade predictions, and real-time portfolio management.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

**Crypto Trading Bot** is an intelligent trading automation system designed for cryptocurrency markets. It combines real-time market data analysis with machine learning models to identify profitable trading opportunities and execute trades automatically.

### Key Capabilities

- **Automated Trading**: Real-time trade execution on Binance
- **ML-Powered Predictions**: XGBoost models for entry/exit signals
- **Technical Analysis**: RSI, MACD, Bollinger Bands, and custom indicators
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

### DevOps & Deployment

- **Containerization**: Docker and Docker Compose
- **Continuous Integration**: GitHub Actions with automated testing
- **Security Scanning**: CodeQL for vulnerability detection
- **Cloud Deployment**: Ready for Google Cloud Platform (Vertex AI, Cloud Run)
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
│  │  └─ Google Cloud Platform (Production)             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Binance API credentials (free account)
- PostgreSQL 13+ (or use Docker)

### Local Development

```bash
# Clone repository
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot.git
cd MyTradingProject-crypto_trading_bot

# Create virtual environment
uv venv
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
|-----------|-----------|---------|
| Framework | FastAPI | REST API and async task handling |
| Language | Python 3.12 | Core trading logic |
| Async Runtime | AsyncIO | Non-blocking I/O operations |
| Task Scheduling | APScheduler | Recurring trading iterations |

### Machine Learning

| Component | Technology | Purpose |
|-----------|------------|---------|
| ML Framework | XGBoost | Trade signal prediction |
| Data Processing | Pandas, NumPy | Feature engineering |
| Technical Analysis | TA | Indicator calculations |
| Experiment Tracking | MLflow | Model versioning and monitoring |
| Data Science | Scikit-learn | Preprocessing and metrics |

### Data & Database

| Component | Technology | Purpose |
|-----------|------------|---------|
| Primary DB | PostgreSQL 16 | Trade history and models |
| ORM | SQLAlchemy 2.0 | Database abstraction |
| Migrations | Alembic | Schema versioning |

### Frontend & Monitoring

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dashboard | Streamlit | Real-time visualization |
| Charting | Plotly | Interactive performance charts |
| Notifications | Telegram Bot API | Real-time alerts |

### DevOps & Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|----------|
| Containerization | Docker & Docker Compose | Local development and deployment |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Code Quality | Ruff, MyPy | Linting and type checking |
| Testing | Pytest | Unit and integration tests |
| Security | CodeQL | Vulnerability scanning |
| Cloud | Google Cloud Platform | Production deployment |

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
│   │   ├── build-and-test.yml     # CI pipeline
│   │   └── codeql.yml             # Security scanning
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
uv run python -m mypy src/

# All checks
uv run python -m pytest && uv run ruff format . && uv run mypy src/
```

### Test Coverage Requirements

- **Minimum**: 70% code coverage
- **Target**: 85%+ code coverage
- **Critical paths**: 90%+ coverage

Current coverage: **45%+** (improving with each contribution)

---

## 🚢 Deployment

### Google Cloud Platform (Recommended)

The bot is optimized for GCP deployment with:

- **Cloud Run**: Serverless container execution
- **Cloud SQL**: Managed PostgreSQL database
- **Vertex AI**: ML model management and monitoring
- **Secret Manager**: Secure credential storage
- **Cloud Build**: Automated CI/CD pipeline

[Full GCP Deployment Guide](GCP_DEPLOYMENT_GUIDE.md)

### Docker Production Build

```bash
# Build production image
docker build -t crypto-trading-bot:latest -f Dockerfile --target api .

# Push to registry
docker push your-registry/crypto-trading-bot:latest
```

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
- **API keys**: Stored in Secret Manager (production)
- **CodeQL scanning**: Automated vulnerability detection
- **Type safety**: Full type hints with mypy checking
- **Input validation**: Pydantic models for all inputs

---

## 📈 Performance Metrics

- **Test Coverage**: 45%+ and growing
- **API Response Time**: < 200ms average
- **Trade Execution**: Real-time (≤500ms order placement)
- **Data Processing**: 100+ instruments/minute
- **Model Inference**: < 50ms per prediction

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup instructions
- Coding standards and conventions
- Testing requirements
- Pull request process
- Community guidelines

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Binance API**: Market data and order execution
- **XGBoost**: Machine learning predictions
- **FastAPI**: Modern web framework
- **Streamlit**: Rapid dashboard development
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

---

**Made with ❤️ by the Crypto Trading Bot Team**

⭐ If you find this useful, please star the repository!