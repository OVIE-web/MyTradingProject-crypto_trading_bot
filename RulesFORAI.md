# Rules for AI Interaction (`RulesFORAI.md`)

## 🚀 Current Project Stage: Live Trading Integration (Phase 3)

This document outlines the rules and standards for interacting with the **MyTradingProject-crypto_trading_bot** codebase.

---

### 📂 Project Structure

The project follows a modular architecture using the `src/` layout:

- **`src/`**: Core source code.
  - `main.py`: **Entry Point**. Orchestrates both Backtesting and Live Trading modes.
  - `data_loader.py`: Fetches and preprocesses OHLCV data from Binance.
  - `feature_engineer.py`: Calculates technical indicators (RSI, MACD, ATR, etc.).
  - `model_manager.py`: Manages XGBoost model training, saving, loading, and prediction.
  - `backtester.py`: Runs strategy simulations on historical data.
  - `notifier.py`: Handles Telegram and Email notifications.
  - `db.py`: Manages PostgreSQL database connections and sessions.
  - `binance_manager.py`: Interfaces with CCXT/Binance API for live market data and order execution.
  - `config.py`: Centralized configuration (loads from `.env`).

- **`data/`**: Stores historical market data (CSV/Pickle) and trained models (`.json`/`.pkl`).
- **`tests/`**: Contains all unit and integration tests (`pytest`).
- **`scripts/`**: Utility scripts for database setup, maintenance, etc.
- **`docker-compose.yml`**: Defines services (PostgreSQL, App).

---

### 🛠️ Tech Stack & Standards

**Core Technologies:**

- **Language**: Python 3.12+ (Managed via `uv`)
- **Web Framework**: FastAPI (for future API endpoints/dashboard)
- **Database**: PostgreSQL 16 + SQLAlchemy ORM (Async support)
- **ML Engine**: XGBoost + Scikit-learn + Pandas
- **Testing**: Pytest (with `pytest-asyncio`, `pytest-mock`)
- **Containerization**: Docker & Docker Compose

**Code Style Rules:**

1. **Type Hinting**: STRICT. All functions must have type hints (`def foo(a: int) -> str:`).
2. **Docstrings**: Google-style docstrings for all modules, classes, and complex functions.
3. **Linting**: Follow `ruff` or `flake8` standards. (PEP 8 compliant).
4. **Async/Await**: Use `async def` for I/O bound operations (DB, API calls).
5. **Error Handling**: Use `try/except` blocks with specific exceptions. Log errors via `logging`, do not just `print`.

---

### ⚙️ Development Workflow

1. **Dependency Management (`uv`):**
    - Add dependency: `uv pip install <package>`
    - Update requirements: `uv pip compile pyproject.toml -o requirements.txt` (or manually maintain `requirements.txt`).
    - Sync environment: `uv pip sync requirements.txt`

2. **Testing:**
    - Run all tests: `pytest`
    - Run specific test file: `pytest tests/test_model.py`
    - Run with coverage: `pytest --cov=src`

3. **Database Migrations:**
    - (If using Alembic in future) `alembic revision --autogenerate -m "message"`
    - Current: Ensure `init_db()` in `src/db.py` is called to create tables.

4. **Docker:**
    - Start services: `docker-compose up -d`
    - Rebuild: `docker-compose up -d --build`
    - View logs: `docker-compose logs -f`

---

### 🤖 Trading & ML Specifics

1. **Model Versioning:**
    - Models are saved with timestamps or version tags in `data/models/`.
    - `model_manager.py` handles loading the latest valid model.

2. **Risk Management:**
    - **Stop-Loss / Take-Profit**: Must be calculated *before* placing orders.
    - **Position Sizing**: Adhere to `TRADE_QUANTITY` in `config.py`.
    - **Live Safety**: `main.py` includes a global exception handler to catch crashes and notify via Telegram.

3. **Data Consistency:**
    - `data_loader.py` fetches historical data.
    - `binance_manager.py` fetches live data.
    - **CRITICAL**: Ensure feature engineering logic in `feature_engineer.py` is IDENTICAL for both historical (training) and live (inference) data to avoid skew.

---

### 🔐 Security & Secrets

- **`.env` File**: NEVER commit `.env`. Use `.env.example` as a template.
- **API Keys**: stored in `.env` (BINANCE_API_KEY, BINANCE_SECRET_KEY).
- **Database Credentials**: stored in `.env` (POSTGRES_USER, POSTGRES_PASSWORD).

---

### 📝 Contribution Checklist

- [ ] Run `pytest` to ensure no regressions.
- [ ] Update `requirements.txt` if adding libs.
- [ ] Update `README.md` or `BeautyOfTheStruggle.md` if architecture changes.
- [ ] Format code (e.g., ruff check . --fix`).
