# 🤖 OvieX Quant Engine Trading System

<p>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white"></a>
  <a href="https://fastapi.tiangolo.com/"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white"></a>
  <a href="https://docs.pydantic.dev/"><img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-2.x-E92063?logo=pydantic&logoColor=white"></a>
  <a href="https://www.postgresql.org/"><img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white"></a>
  <a href="https://redis.io/"><img alt="Redis" src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white"></a>
  <a href="https://docs.celeryq.dev/"><img alt="Celery" src="https://img.shields.io/badge/Celery-5.6.3-37814A?logo=celery&logoColor=white"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1.2.0-1C3C3C"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/Ruff-linting-261230?logo=ruff&logoColor=white"></a>
  <a href="https://mypy-lang.org/"><img alt="MyPy" src="https://img.shields.io/badge/Type%20Checking-MyPy-2A6DB2"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

A modular, production-ready cryptocurrency trading system with a FastAPI backend, ML-powered trade predictions, LangGraph-based trading-agent orchestration, risk guardrails, market-data scraping, real-time workers, and portfolio monitoring.

The project has been refactored from the old `src.*` layout into a cleaner `app.*` architecture with separated API routes, domain logic, guardrails, services, workers, agents, tools, schemas, models, scrapers, and utilities.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Security](#-security)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## 🎯 Overview

OvieX Quant Engine is an intelligent trading automation system designed for cryptocurrency markets. It combines Binance market data, technical analysis, XGBoost inference, deterministic strategy logic, guardrail enforcement, and optional AI-agent reasoning to support safer trading decisions.

### Key Capabilities

- Automated trading workflow with Binance market-data and order-service integration.
- ML-powered predictions using XGBoost models and model metadata.
- Technical analysis with RSI, moving averages, Bollinger Bands, momentum, and ATR.
- Risk management with position sizing, drawdown checks, stop-loss, and take-profit logic.
- Guardrails for trade limits, cooldowns, market-data freshness, and safety checks.
- FastAPI backend with authentication, predictions, trades, webhooks, and health checks.
- Streamlit dashboard for local visualization and monitoring.
- LangGraph trading agent for market summaries, signal explanations, approval routing, and final execution gating.
- PostgreSQL persistence with SQLAlchemy models and Alembic schema migrations.
- Celery and Redis support for background trading tasks and periodic scheduling.

## ✨ Features

### Core Trading Features

#### Intelligent Signal Generation

- ML-based entry/exit signal predictions.
- Domain signal normalization for buy, sell, and hold decisions.
- Confidence-weighted signal combination.
- Technical indicator feature pipeline.
- Strategy decisions that can block duplicate entries or unsafe sells.

#### Trade Execution

- Market order execution through the Binance service.
- Offline/mock mode when Binance credentials are not configured.
- Risk-aware trade execution service.
- Database persistence for executed trades.
- Failure handling with rollback support.

#### Portfolio and Risk Management

- Account state modeling for available cash, equity, and positions.
- Position sizing based on configured risk per trade.
- Stop-loss and take-profit price generation.
- Daily loss and drawdown limit checks.
- Trade limit enforcement for symbols, actions, notional exposure, and cooldowns.

### AI Agent Layer

#### LangGraph Trading Agent

- Summarizes market context.
- Explains the model signal in human-readable language.
- Proposes a strategy action.
- Routes risky trades to human approval.
- Applies guardrails before final execution.
- Calls the trade execution service only after approval.

#### Future RAG and Knowledge Workflows

- `app/vector_store/` is reserved for embeddings, strategy notes, market reports, trading journals, and research memory.
- LlamaIndex-style RAG can be added later for document search and market-research workflows.
- The main trading orchestrator should remain LangGraph because the trading flow is stateful and approval-driven.

### API and Integration

#### REST API

- Health checks.
- Token-based authentication.
- Current-user endpoint.
- Prediction endpoint.
- Model reload endpoint.
- Trade create, list, retrieve, and delete endpoints.
- GitHub-style webhook receiver.

#### Dashboard

- Streamlit frontend for model loading, prediction workflows, charts, and local monitoring.
- Plotly visualizer helpers for candlesticks, indicators, trades, and portfolio curves.

### DevOps and Quality

- Docker and Docker Compose for local orchestration.
- PostgreSQL 16 for persistence.
- Redis 7 for Celery broker/result backend.
- Celery worker and beat profiles for async/periodic execution.
- Alembic migrations for schema versioning.
- Ruff linting and formatting.
- MyPy type checking.
- Pytest coverage across mirrored `tests/` subfolders.
- Archived experimental enhancement scripts for historical reference.

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│              OvieX Quant Engine Trading System              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Client Layer                            │   │
│  │  ├─ Streamlit Dashboard (Port 8501)                  │   │
│  │  ├─ FastAPI REST API (Port 8000)                     │   │
│  │  ├─ Swagger UI (/docs)                               │   │
│  │  └─ Health Checks and Monitoring                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Agent and API Orchestration             │   │
│  │  ├─ LangGraph Trading Agent                          │   │
│  │  ├─ Market Context Summary                           │   │
│  │  ├─ Model Signal Explanation                         │   │
│  │  ├─ Human Approval Routing                           │   │
│  │  └─ Guardrail-Gated Execution                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Core Trading Engine                     │   │
│  │  ├─ Market Data Scraper and Binance Service          │   │
│  │  ├─ Technical Indicator Calculator                   │   │
│  │  ├─ Feature Engineering Pipeline                     │   │
│  │  ├─ ML Model Inference (XGBoost)                     │   │
│  │  ├─ Signal Generator and Strategy Logic              │   │
│  │  ├─ Risk Manager and Position Sizer                  │   │
│  │  ├─ Trade Limits and Safety Checks                   │   │
│  │  └─ Trade Execution Service                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Data and Persistence Layer              │   │
│  │  ├─ PostgreSQL (Trades, Predictions, Users)          │   │
│  │  ├─ SQLAlchemy ORM Models                            │   │
│  │  ├─ Alembic Schema Migrations                        │   │
│  │  ├─ Model Registry and XGBoost Artifacts             │   │
│  │  └─ Optional Vector Store for RAG Memory             │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              External Services                       │   │
│  │  ├─ Binance API (Market Data and Orders)             │   │
│  │  ├─ Telegram Bot API (Notifications)                 │   │
│  │  ├─ SMTP Email Alerts                                │   │
│  │  └─ GitHub Webhooks                                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- PostgreSQL 13+ if running without Docker
- Redis if running Celery without Docker
- Binance API credentials for live/testnet trading

### Local Development

```powershell
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot.git
cd MyTradingProject-crypto_trading_bot

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install uv
uv pip install -e ".[dev]"
```

Create and edit your environment file:

```powershell
Copy-Item .env.example .env
```

Run database initialization:

```powershell
.\.venv\Scripts\python.exe -m scripts.init_database
```

Start the API:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
Swagger UI: http://127.0.0.1:8000/docs
ReDoc:      http://127.0.0.1:8000/redoc
Health:     http://127.0.0.1:8000/health
```

### Docker Quick Start

```powershell
docker compose up --build
```

Run dashboard services:

```powershell
docker compose --profile dashboard up --build
```

Run the direct scheduler:

```powershell
docker compose --profile scheduler up --build
```

Run Celery worker and beat:

```powershell
docker compose --profile celery up --build
```

Access services:

```text
FastAPI Docs: http://localhost:8000/docs
Streamlit:    http://localhost:8501
PgAdmin:      http://localhost:5050
```

## 💻 Technology Stack

### Backend and Core

| Component | Technology | Purpose |
| --- | --- | --- |
| Framework | FastAPI | REST API, auth endpoints, and router layer |
| Language | Python 3.12 | Core trading logic |
| Validation | Pydantic 2 | Request and response schemas |
| Async Runtime | AsyncIO | Non-blocking worker and scheduler flows |
| Agent Orchestration | LangGraph | Stateful trading-agent workflow |

### Machine Learning

| Component | Technology | Purpose |
| --- | --- | --- |
| ML Framework | XGBoost | Trade signal prediction |
| Data Processing | Pandas, NumPy | Feature engineering and data preparation |
| Technical Analysis | `ta` | RSI, Bollinger Bands, SMA, ATR |
| Experiment Tracking | MLflow | Model and experiment tracking |
| Data Science | Scikit-learn, imbalanced-learn | Preprocessing and metrics |

### Data and Database

| Component | Technology | Purpose |
| --- | --- | --- |
| Primary DB | PostgreSQL 16 | Trades, predictions, users, model metadata |
| ORM | SQLAlchemy 2 | Database abstraction |
| Migrations | Alembic | Schema versioning |
| Queue Backend | Redis 7 | Celery broker and result backend |

### Frontend and Monitoring

| Component | Technology | Purpose |
| --- | --- | --- |
| Dashboard | Streamlit | Local monitoring and prediction workflow |
| Charting | Plotly | Interactive charts and trade visualization |
| Notifications | Telegram Bot API, SMTP | Trading alerts |
| Logs | Python logging | API, worker, and service diagnostics |

### DevOps and Infrastructure

| Component | Technology | Purpose |
| --- | --- | --- |
| Containerization | Docker, Docker Compose | Local development and deployment |
| Background Tasks | Celery, Celery Beat | Async and periodic trading jobs |
| Code Quality | Ruff | Linting and formatting |
| Type Safety | MyPy | Static type checking |
| Testing | Pytest | Unit and integration tests |

## 📁 Project Structure

```text
crypto-trading-bot/
├── app/
│   ├── agents/                  # LangGraph trading agent and agent schemas/tools
│   ├── api/
│   │   └── routes/              # Health, predictions, trades, webhook routers
│   ├── core/                    # Config, settings, security, logging, env loading
│   ├── db/                      # Engine, sessions, metadata, init helpers
│   ├── domain/                  # Trading strategy, signals, risk management
│   ├── frontend/                # Streamlit dashboard
│   ├── guardrails/              # Safety checks and trade limits
│   ├── middleware/              # Error handling and request logging
│   ├── models/                  # SQLAlchemy models: trade, prediction, user
│   ├── schemas/                 # Pydantic schemas for auth, trades, predictions
│   ├── scrapers/                # Market data scraper and OHLCV normalization
│   ├── services/                # Binance, model, training, execution, notification services
│   ├── tools/                   # Indicators, feature engineering, diagnostics, visualization
│   ├── utils/                   # Shared helpers and validators
│   ├── vector_store/            # Reserved for future RAG/vector memory workflows
│   ├── workers/                 # Bot runner, direct scheduler, Celery worker/tasks
│   ├── main.py                  # FastAPI application entry point
│   └── py.typed                 # Type marker for the app package
│
├── tests/
│   ├── agents/                  # Agent workflow tests
│   ├── api/                     # API and router tests
│   ├── core/                    # Config, logging, security tests
│   ├── db/                      # Database/session/init tests
│   ├── domain/                  # Strategy, signal, risk tests
│   ├── frontend/                # Streamlit helper tests
│   ├── guardrails/              # Trade limit and safety tests
│   ├── middleware/              # Error/request middleware tests
│   ├── models/                  # SQLAlchemy model tests
│   ├── schemas/                 # Pydantic schema tests
│   ├── scrapers/                # Market data scraper tests
│   ├── services/                # Service and notifier tests
│   ├── tools/                   # Feature, diagnostics, visualizer tests
│   ├── utils/                   # Helper and validator tests
│   └── workers/                 # Bot runner, scheduler, Celery tests
│
├── alembic/
│   ├── versions/                # Database migration revisions
│   ├── env.py                   # Alembic migration environment
│   ├── script.py.mako           # Migration file template
│   └── README.md                # Migration workflow notes
│
├── docker/
│   └── entrypoints/             # API, worker, and beat entrypoints
├── models/                      # XGBoost model artifacts and metadata
├── scripts/
│   ├── archive/                 # Archived experimental scripts
│   ├── init_database.py         # Create missing DB tables from metadata
│   └── smoke_notify.py          # Notification smoke test helper
├── Dockerfile                   # Multi-target image build
├── docker-compose.yml           # Local service orchestration
├── pyproject.toml               # Metadata, dependencies, Ruff, MyPy, coverage
├── pytest.ini                   # Active pytest configuration
├── RulesFORAI.md                # AI collaboration and system rules
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## 🛠️ Development

### Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Run a focused test folder:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\scrapers -q --no-cov
```

Run tests with coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Open the HTML coverage report:

```text
htmlcov/index.html
```

### Code Quality

```powershell
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m ruff format app tests
.\.venv\Scripts\python.exe -m mypy app tests --config-file mypy.ini --no-incremental
```

Focused checks for one package:

```powershell
.\.venv\Scripts\python.exe -m ruff check app\scrapers tests\scrapers
.\.venv\Scripts\python.exe -m mypy app\scrapers tests\scrapers --python-version 3.12 --ignore-missing-imports --strict-optional --no-incremental --show-error-codes --follow-imports=skip
.\.venv\Scripts\python.exe -m pytest tests\scrapers -q --no-cov
```

### Database and Migrations

Initialize tables from SQLAlchemy metadata:

```powershell
.\.venv\Scripts\python.exe -m scripts.init_database
```

Apply Alembic migrations:

```powershell
alembic upgrade head
```

Create a new migration:

```powershell
alembic revision --autogenerate -m "describe_change"
```

### Running Workers

Direct scheduler:

```powershell
.\.venv\Scripts\python.exe -m app.workers.scheduler
```

Celery worker:

```powershell
celery -A app.workers.celery_worker:celery_app worker --loglevel=info
```

Celery beat:

```powershell
celery -A app.workers.celery_worker:celery_app beat --loglevel=info
```

Do not run both the direct scheduler and Celery beat against live trading unless you intentionally want multiple trading loops.

## 📊 Example Workflow

```python
from app.domain.risk_management import AccountState, RiskConfig, review_trade
from app.domain.trading_strategy import StrategyConfig, decide_from_model_signal
from app.guardrails.trade_limits import TradeLimitConfig, check_trade_limits
from app.scrapers.market_data_scraper import MarketDataScrapeRequest, scrape_latest_ohlcv
from app.tools.feature_engineer import calculate_technical_indicators

# 1. Fetch and normalize market data.
market_data = scrape_latest_ohlcv(
    MarketDataScrapeRequest(symbol="BTCUSDT", interval="4h", limit=500)
).data

# 2. Build model-ready features.
features = calculate_technical_indicators(market_data)

# 3. Convert a model signal into a strategy decision.
decision = decide_from_model_signal(
    signal=1,
    confidence=0.92,
    current_position=0,
    config=StrategyConfig(),
)

# 4. Review risk before execution.
risk_decision = review_trade(
    decision,
    account=AccountState(cash=10_000, equity=10_000, current_position=0),
    price=float(features["close"].iloc[-1]),
    config=RiskConfig(),
)

# 5. Apply final trade limits and guardrails.
limits = check_trade_limits(
    risk_decision,
    config=TradeLimitConfig(),
)
```

## 🔒 Security

- Keep all secrets in `.env` or a secure secret manager.
- Never commit Binance API keys, JWT secrets, database passwords, or SMTP credentials.
- Use Binance testnet before any live-trading flow.
- Use least-privilege API keys and disable withdrawals on exchange keys.
- Rotate credentials regularly.
- Validate all API inputs through Pydantic schemas.
- Keep live execution gated by risk checks, trade limits, and safety checks.
- Use human approval for risky agent-proposed trades.

## 📈 Quality Baseline

- Test framework: Pytest.
- Current suite shape: mirrored test subfolders matching `app/`.
- Recent local baseline: 400+ tests passing with coverage above 85%.
- Quality gates: Ruff linting, Ruff formatting, MyPy type checking, and pytest coverage.
- Critical paths: risk management, trade limits, execution service, model prediction, API routers, and workers.

## 🗺️ Roadmap

- [ ] Multi-pair concurrent trading.
- [ ] Portfolio-level risk management.
- [ ] Real-time model retraining workflow.
- [ ] RAG-backed research agent using `vector_store`.
- [ ] Trading journal search and strategy-note retrieval.
- [ ] Advanced charting and analytics.
- [ ] Parallelized backtesting optimization.
- [ ] Additional exchange integrations such as Kraken and Coinbase.
- [ ] Cloud deployment guides for AWS, Azure, and GCP.

## 🤝 Contributing

Contributions are welcome. Please follow:

- `CONTRIBUTING.md` for the development workflow.
- `CODE_OF_CONDUCT.md` for community standards.
- `RulesFORAI.md` for AI-assisted changes in this repository.

Before opening a pull request, run:

```powershell
.\.venv\Scripts\python.exe -m ruff check app tests
.\.venv\Scripts\python.exe -m mypy app tests --config-file mypy.ini --no-incremental
.\.venv\Scripts\python.exe -m pytest tests
```

## 📝 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Binance API for market data and exchange integration.
- XGBoost for machine-learning prediction support.
- FastAPI for the backend framework.
- Streamlit and Plotly for local dashboarding and visualization.
- PostgreSQL, SQLAlchemy, and Alembic for persistence and migrations.
- LangGraph for stateful agent orchestration.
- The open-source Python community for the tools that make this system possible.

## 📞 Support and Contact

- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Email: oviemsaniyo30@gmail.com

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. Cryptocurrency trading involves significant financial risk. Past performance does not guarantee future results.

Always:

- Start in testnet or offline/mock mode.
- Use proper risk management.
- Never trade with funds you cannot afford to lose.
- Thoroughly test strategies before live trading.
- Keep API keys secure and rotated regularly.

Made with ❤️ by the OvieX Quant Engine team.

⭐ If you find this useful, please star the repository.
