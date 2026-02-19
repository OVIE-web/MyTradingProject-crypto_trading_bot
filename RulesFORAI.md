# Rules for AI Interaction (`RulesFORAI.md`)

## 🚀 Current Project Stage: Production-Ready Crypto Trading Bot (Phase 3 - Live Trading)

This document outlines the rules, standards, and best practices for interacting with the **MyTradingProject-crypto_trading_bot** codebase. All contributors and AI assistants must follow these guidelines.

---

## 📂 Project Structure

The project uses a **modular, production-grade architecture** with the `src/` layout:

``
crypto-trading-bot/
├── src/                          # Core application source code
│   ├── main_api.py              # FastAPI application & REST endpoints
│   ├── bot_runner.py            # Main trading loop orchestrator (async)
│   ├── backtester.py            # Strategy backtesting engine
│   ├── binance_manager.py       # Binance API integration (OHLCV, orders)
│   ├── data_loader.py           # Market data fetching & preprocessing
│   ├── feature_engineer.py      # Technical indicator calculations (RSI, MACD, etc.)
│   ├── model_manager.py         # ML model training, inference, versioning
│   ├── notification.py          # Email & Telegram alert system
│   ├── notifier.py              # Notification orchestration
│   ├── db.py                    # PostgreSQL ORM models & sessions (SQLAlchemy)
│   ├── settings.py              # Environment configuration & validation
│   ├── streamlit_app.py         # Real-time dashboard frontend
│   └── routers/
│       ├── predict.py           # ML prediction endpoints
│       └── trades.py            # Trade history endpoints
│
├── tests/                        # Comprehensive test suite (pytest)
│   ├── test_settings.py         # Configuration validation tests
│   ├── test_binance_manager.py  # Exchange integration tests
│   ├── test_backtester.py       # Strategy testing
│   ├── test_model_manager.py    # ML pipeline tests
│   ├── test_feature_engineer.py # Indicator calculation tests
│   ├── test_data_loader.py      # Data processing tests
│   └── notifications/           # Alert system tests
│
├── .github/
│   ├── workflows/
│   │   ├── .github/test.yml     # CI/CD pipeline (pytest, mypy, ruff)
│   │   └── dependabot.yml       # Automated dependency updates
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml       # Bug report form
│       └── feature_request.yml  # Feature request form
│
├── scripts/
│   ├── init_database.py         # Database initialization
│   └── wait_for_postgres.py     # Connection polling
│
├── Dockerfile                    # Multi-stage build (api, bot, streamlit, test targets)
├── docker-compose.yml           # Local development environment
├── pyproject.toml              # Project metadata & dependencies
├── pytest.ini                  # Test configuration
├── CODE_OF_CONDUCT.md          # Community standards
├── CONTRIBUTING.md             # Contribution guidelines
├── dependabot.yml              # Automated updates config
└── README.md                   # Project documentation

---

## 🛠️ Tech Stack & Standards

### Core Technologies

| Component | Technology | Version |
| --------- | ---------- | ------- |
| **Language** | Python | 3.12+ |
| **Package Manager** | UV | Latest |
| **Web Framework** | FastAPI | 0.111+ |
| **Async Runtime** | AsyncIO | Built-in |
| **Database** | PostgreSQL | 16+ |
| **ORM** | SQLAlchemy | 2.0+ |
| **ML Framework** | XGBoost | 3.0+ |
| **Data Processing** | Pandas, NumPy | Latest stable |
| **Testing** | Pytest | 9.0+ |
| **Linting** | Ruff | 0.8+ |
| **Type Checking** | MyPy | 1.10+ |
| **Containerization** | Docker & Docker Compose | Latest |
| **Dashboard** | Streamlit | 1.28+ |
| **Exchange API** | Python-Binance | 1.0+ |

### Code Style Standards

**1. Type Hinting (MANDATORY):**

```python
# ✅ CORRECT
def calculate_position_size(
    balance: float,
    risk_percent: float
) -> float:
    """Calculate position size based on risk percentage."""
    return balance * (risk_percent / 100)

# ❌ WRONG
def calculate_position_size(balance, risk_percent):
    return balance * (risk_percent / 100)
```

**2. Docstrings (Google-style, REQUIRED):**

```python
def fetch_market_data(symbol: str, interval: str = "1h") -> dict:
    """Fetch OHLCV data from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Timeframe ('1h', '4h', '1d', etc.)
        
    Returns:
        Dictionary containing OHLCV data and metadata
        
    Raises:
        BinanceAPIError: If exchange is unavailable
        ValueError: If symbol is invalid
        
    Example:
        >>> data = await fetch_market_data('BTCUSDT', '1h')
        >>> len(data['closes']) > 0
        True
    """
    pass
```

**3. Async/Await (REQUIRED for I/O operations):**

```python
# ✅ CORRECT - API calls are async
async def place_order(symbol: str, side: str, quantity: float) -> dict:
    """Place market order asynchronously."""
    return await self.binance_client.place_market_order(symbol, side, quantity)

# ❌ WRONG - Blocking call in async context
async def place_order(symbol: str, side: str, quantity: float) -> dict:
    return self.binance_client.place_market_order(symbol, side, quantity)  # Blocks!
```

**4. Error Handling (Specific exceptions, REQUIRED):**

```python
# ✅ CORRECT
try:
    balance = await binance_manager.get_account_balance()
except BinanceAPIError as e:
    logger.error(f"Failed to fetch balance: {e}")
    await notifier.send_alert(f"Balance fetch failed: {e}")
    raise
except Exception as e:
    logger.critical(f"Unexpected error: {e}")
    raise

# ❌ WRONG
try:
    balance = await binance_manager.get_account_balance()
except:
    print("Error!")
```

**5. Logging (NOT print statements):**

```python
# ✅ CORRECT
import logging
logger = logging.getLogger(__name__)

logger.info(f"Trading signal: {signal} for {symbol}")
logger.warning(f"Low balance: ${balance}")
logger.error(f"Order failed: {error}")

# ❌ WRONG
print(f"Trading signal: {signal}")
print("Low balance!")
```

**6. PEP 8 Compliance:**

- Line length: Maximum 100 characters (per `ruff` config)
- Indentation: 4 spaces (never tabs)
- Naming: `snake_case` for functions/variables, `PascalCase` for classes
- Imports: Group stdlib, third-party, local (alphabetically)

---

## ⚙️ Development Workflow

### 1. Dependency Management (UV)

```bash
# Add a new dependency
uv pip install package-name

# Install development dependencies
uv pip install -e ".[dev]"

# Verify dependencies compile
uv pip compile pyproject.toml

# Sync environment with lock file
uv pip sync requirements.txt
```

**Rules:**

- ✅ Always update `pyproject.toml` (not `requirements.txt` directly)
- ✅ Pin versions for production dependencies
- ✅ Use version ranges for dev dependencies (`>=1.0,<2.0`)
- ❌ Never commit `requirements.txt` if managing via `pyproject.toml`

### 2. Testing (Pytest - MANDATORY)

```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest tests/test_model_manager.py -v

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_settings.py::test_get_env_missing_required -v
```

**Requirements:**

- ✅ All new features must have tests
- ✅ Coverage target: **70%+ minimum**, **85%+ target**
- ✅ All tests must pass before pushing
- ✅ Use `pytest-mock` for mocking external APIs
- ❌ No untested production code

### 3. Code Quality Checks

```bash
# Format code with ruff
uv run ruff format src/ tests/

# Check for linting issues
uv run ruff check src/ tests/

# Type checking with mypy
uv run mypy src/

# Run all checks
uv run pytest && uv run ruff format . && uv run mypy src/
```

**Pre-commit:**

```bash
# All of these MUST pass before git push
uv run pytest -v                    # 59+ tests passing
uv run ruff format src/ tests/      # Code formatted
uv run ruff check src/ tests/       # No linting errors
uv run mypy src/                    # No type errors
```

### 4. Docker & Local Development

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down

# Clean up volumes (WARNING: deletes data)
docker-compose down -v

# Run tests in Docker
docker-compose run --rm test
```

**Services:**

- `db` (PostgreSQL 16) - Port 5432
- `api` (FastAPI) - Port 8000
- `streamlit` (Dashboard) - Port 8501
- `bot` (Trading worker)
- `test` (Pytest runner)

### 5. Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-indicator

# Commit with conventional format
git commit -m "feat: add MACD indicator calculation

- Implement MACD calculation in feature_engineer.py
- Add comprehensive tests for edge cases
- Update technical_indicators() function"

# Push and create PR
git push origin feature/new-indicator

# PR checks must pass:
# ✅ CI/CD pipeline (pytest, mypy, ruff)
# ✅ Code coverage maintained
# ✅ All tests passing
```

**Commit Message Format (Conventional Commits):**

```bash
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `ci`, `chore`

---

## 🤖 Trading & ML Specifics

### 1. Feature Engineering Consistency (CRITICAL)

### Rule: Training and inference MUST use identical logic

```python
# ✅ CORRECT - Same function for both
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators for both training and live inference."""
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd(df['close'])['macd_line']
    return df

# Training
train_df = await load_historical_data(symbol, start_date, end_date)
train_features = calculate_indicators(train_df)

# Live
live_df = await fetch_market_data(symbol, interval)
live_features = calculate_indicators(live_df)  # SAME FUNCTION!
```

### 2. Model Versioning

```python
# Model storage: data/models/
# Naming: <symbol>_<version>_<timestamp>.json
# Example: BTCUSDT_v1.0_2026-02-19.json

# Latest model loading:
model = model_manager.load_latest_model(symbol)

# Version tracking in database for audit
```

### 3. Risk Management (MANDATORY)

```python
# All orders must include:
position_size = calculate_position_size(
    balance=account_balance,
    risk_percent=2.0  # Max 2% per trade
)

stop_loss = current_price * 0.98  # 2% below entry
take_profit = current_price * 1.05  # 5% above entry

# NEVER place order without SL/TP validation
assert position_size > 0, "Invalid position size"
assert stop_loss < current_price, "Invalid stop loss"
assert take_profit > current_price, "Invalid take profit"
```

### 4. Data Consistency

**CRITICAL: Feature engineering must be identical for:**

1. Historical data (training/backtesting)
2. Live data (inference)

If they differ, the model will have **prediction skew** and perform poorly.

```python
# BAD - Different logic
def train_features(df):
    return df.rsi  # Missing normalization

def inference_features(df):
    return (df.rsi - 50) / 25  # Normalized

# GOOD - Shared function
def engineer_features(df):
    df['rsi_norm'] = (ta.rsi(df.close) - 50) / 25
    return df
```

---

## 🔐 Security & Secrets Management

### 1. Environment Variables (.env)

```bash
# ✅ CORRECT - .env is in .gitignore
.env (NEVER committed)

# Template for team:
.env.example (ALWAYS committed, no secrets)
```

**Required variables:**

```env
# Exchange
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/crypto_bot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password

# Trading
INITIAL_BALANCE=1000.0
RISK_PERCENT=2.0

# Notifications
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Environment
ENV=development  # or production
DEBUG=false
```

### 2. API Key Rotation

- Rotate keys monthly
- Use read-only keys for data fetching
- Trading keys require 2FA
- Store in `.env` (never in code)

### 3. Database Credentials

- Use strong passwords (20+ chars, mixed case)
- Store in `.env` only
- Different credentials for dev/prod
- Enable SSL for prod connections

---

## 📝 Contribution Checklist

Before submitting a PR, verify ALL of these:

- [ ] **Tests**: Run `uv run pytest -v` → All pass
- [ ] **Coverage**: Run `uv run pytest --cov=src` → 70%+ maintained
- [ ] **Formatting**: Run `uv run ruff format src/ tests/`
- [ ] **Linting**: Run `uv run ruff check src/ tests/` → No errors
- [ ] **Type Checking**: Run `uv run mypy src/` → No errors
- [ ] **Docstrings**: All functions have Google-style docstrings
- [ ] **Type Hints**: All functions have type annotations
- [ ] **Async/Await**: I/O operations use async/await
- [ ] **Logging**: Errors logged (not printed)
- [ ] **Error Handling**: Specific exceptions caught, not bare `except:`
- [ ] **Dependencies**: Added to `pyproject.toml`, not installed locally only
- [ ] **Documentation**: README/CONTRIBUTING updated if needed
- [ ] **Secrets**: No API keys, passwords, or tokens in code
- [ ] **Commit Messages**: Conventional format (`feat:`, `fix:`, etc.)
- [ ] **Feature Branch**: Created from `main`, not working on main directly

---

## 🚨 Common Mistakes (DO NOT DO THESE)

| ❌ Wrong | ✅ Correct |
| ------- | ----------- |
| `def foo(x):` | `def foo(x: int) -> str:` |
| `print("error")` | `logger.error("error")` |
| `try: ... except:` | `try: ... except SpecificError as e:` |
| `requests.get()` (blocking) | `await async_client.get()` (async) |
| API key in code | API key in `.env` |
| No tests | Comprehensive tests with `pytest` |
| `ruff check` skipped | All linting passes |
| `mypy` warnings ignored | Zero mypy errors |
| Function with no docstring | Google-style docstrings required |
| Mixing stdlib/local imports | Grouped, alphabetized imports |

---

## 📞 Questions or Clarifications?

If unclear on any standard:

1. Check this document first
2. Review `CONTRIBUTING.md` for contribution workflow
3. Open an issue with `question` label
4. Ask in PR comments

---

## 📅 Last Updated

**Date**: February 19, 2026
**Version**: 3.0
**Status**: Production-Ready
