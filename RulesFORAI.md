# Rules for AI Interaction (`RulesFORAI.md`)

## Current Project Stage: AI-Powered Crypto Trading Bot

This document defines the rules, standards, and best practices for working with the
**MyTradingProject-crypto_trading_bot** codebase. All contributors and AI assistants must
follow these guidelines.

The system is now organized around a refactored `app/` package, mirrored test folders,
versioned database migrations, typed domain logic, deterministic guardrails, and an
AI-agent layer for trading workflow orchestration and future research/RAG workflows.

---

## Project Structure

The project uses a modular, production-oriented architecture with `app/` as the canonical
application package.

```text
crypto-trading-bot/
├── app/
│   ├── agents/                 # AI agents and agent-safe workflow orchestration
│   │   ├── trading_agent.py     # LangGraph trading workflow orchestration
│   │   ├── research_agent.py    # Future LlamaIndex/RAG research workflow
│   │   ├── agents_schemas.py    # Typed agent state, requests, and results
│   │   └── agents_tools.py      # Deterministic agent tool wrappers
│   ├── api/
│   │   └── routes/              # FastAPI route modules
│   │       ├── health.py
│   │       ├── predictions.py
│   │       ├── trades.py
│   │       └── webhook.py
│   ├── core/                    # Config, env loading, logging, settings, security
│   ├── db/                      # SQLAlchemy engine, sessions, init helpers
│   ├── domain/                  # Trading strategy, risk management, signals
│   ├── frontend/                # Streamlit dashboard
│   ├── guardrails/              # Safety checks and trade limits
│   ├── middleware/              # Request logging and API error handling
│   ├── models/                  # SQLAlchemy ORM models
│   ├── schemas/                 # Pydantic API schemas
│   ├── scrapers/                # Market data ingestion
│   ├── services/                # Binance, model, training, execution, notifications
│   ├── tools/                   # Feature engineering, diagnostics, visualizer
│   ├── utils/                   # Shared helpers and validators
│   ├── vector_store/            # Future embeddings/indexes for research/RAG
│   ├── workers/                 # Celery worker, scheduler, trading runner
│   └── main.py                  # FastAPI app entrypoint
├── alembic/                     # Versioned database migrations
│   ├── env.py                   # Uses app metadata and DATABASE_URL
│   ├── script.py.mako           # Migration file template
│   └── versions/                # Migration revisions
├── docker/
│   └── entrypoints/             # API, worker, and beat startup scripts
├── scripts/
│   ├── init_database.py         # Local schema bootstrap helper
│   ├── smoke_notify.py          # Notification smoke test helper
│   └── archive/                 # Archived experimental scripts
├── tests/                       # Mirrors app/ package structure
│   ├── agents/
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── domain/
│   ├── frontend/
│   ├── guardrails/
│   ├── middleware/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── tools/
│   ├── utils/
│   ├── vector_store/
│   └── workers/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── pytest.ini
├── mypy.ini
├── uv.lock
└── README.md
```

Rules:

- `app/` is the only active application package.
- Do not add new code to `src/`.
- Tests must mirror the `app/` subpackage they validate.
- Runtime business logic belongs in `domain/`, `guardrails/`, or `services/`, not in route handlers.
- AI agents must orchestrate existing deterministic logic instead of duplicating strategy, risk, or execution rules.

---

## Tech Stack & Standards

### Core Technologies

| Component | Technology | Project Standard |
| --------- | ---------- | ---------------- |
| Language | Python | 3.12+ |
| Package Manager | uv | Use `pyproject.toml` and `uv.lock` |
| API | FastAPI, Uvicorn | `app.main:app` |
| Database | PostgreSQL | SQLAlchemy 2.0 ORM |
| Migrations | Alembic | `alembic/versions` |
| Workers | Celery, Redis | Scheduled trading jobs |
| ML | XGBoost, scikit-learn | Model service and training service |
| Data | Pandas, NumPy | Feature and market data pipelines |
| Dashboard | Streamlit | `app/frontend/streamlit_app.py` |
| Exchange | python-binance | Wrapped by services |
| Trading Agent | LangGraph | Stateful trading workflow |
| Research Agent | LlamaIndex | Planned RAG/knowledge workflow |
| Testing | Pytest | Mirrored test structure |
| Linting | Ruff | Required before commit |
| Type Checking | MyPy | Required for touched modules |
| Containers | Docker Compose | API, workers, beat, db, redis |

### Code Style Standards

**1. Type Hinting Is Required**

```python
def calculate_position_size(balance: float, risk_percent: float) -> float:
    """Calculate position size based on risk percentage."""
    return balance * (risk_percent / 100)
```

Do not add untyped production functions unless there is a clear compatibility reason.

**2. Docstrings Are Required For Public APIs**

```python
def fetch_market_data(symbol: str, interval: str = "1h") -> dict[str, object]:
    """Fetch OHLCV data for a trading pair."""
    ...
```

Keep docstrings practical. Explain why the function exists, not obvious line-by-line behavior.

**3. Logging Must Use Project Logging**

```python
from app.core.logging import get_logger

LOG = get_logger(__name__)
LOG.info("Trade decision created. symbol=%s action=%s", symbol, action)
```

Do not use `print()` in production code.

**4. Error Handling Must Be Specific**

```python
try:
    result = service.execute_market_trade(request)
except ValueError as exc:
    LOG.warning("Invalid trade request: %s", exc)
    raise
```

Avoid bare `except:` blocks.

**5. Imports Must Follow Ruff**

- Standard library imports first.
- Third-party imports second.
- First-party `app.*` imports last.
- Run Ruff before committing.

---

## Development Workflow

### 1. Dependency Management

Use `uv` and keep dependency changes reproducible.

```powershell
uv add "package==version"
uv add --dev "package==version"
uv sync --all-extras
```

Rules:

- Add runtime dependencies to `pyproject.toml`.
- Keep `uv.lock` updated when dependencies change.
- Do not rely on packages installed only in `.venv`.
- Pin production dependencies unless there is a deliberate reason not to.
- If a dependency is experimental, prefer a dev extra or archived workflow.

### 2. Testing

Run tests with the refactored layout.

```powershell
python -m pytest tests -q
python -m pytest tests\agents -q --no-cov
python -m pytest tests\api -q --no-cov
```

When Windows temp permissions interfere, use a workspace temp base.

```powershell
python -m pytest tests -q --basetemp=.pytest-tmp
```

Rules:

- Every new production module needs tests.
- Tests should live in the mirrored folder under `tests/`.
- Avoid real exchange calls, live DB calls, and network calls in unit tests.
- Use dependency overrides, fakes, or mocks for external systems.
- Do not weaken tests to hide real bugs.

### 3. Code Quality Checks

```powershell
python -m ruff check app tests scripts alembic
python -m pytest tests -q
python -m mypy app\agents tests\agents --python-version 3.12 --ignore-missing-imports --strict-optional --no-incremental --show-error-codes --follow-imports=skip
```

Rules:

- Run Ruff on changed Python paths.
- Run focused Pytest for changed modules.
- Run focused MyPy for changed typed modules.
- Full-project MyPy may be slower; focused MyPy is acceptable while refactoring, but do not ignore real type errors.

### 4. Database Workflow

Use Alembic for versioned schema changes.

```powershell
python -m alembic -c alembic.ini heads
python -m alembic -c alembic.ini history
python -m alembic -c alembic.ini revision --autogenerate -m "describe change"
python -m alembic -c alembic.ini upgrade head
```

Use `scripts/init_database.py` only as a local bootstrap helper.

```powershell
python -m scripts.init_database
```

Rules:

- Use Alembic for production schema changes.
- Do not edit applied migration files unless the migration has not been shared.
- If tables already exist before Alembic, use `alembic stamp head` to baseline that database.
- Model changes must include a migration and tests.

### 5. Docker & Workers

```powershell
docker compose up -d --build
docker compose logs -f api
docker compose logs -f worker
docker compose down
```

Rules:

- API startup belongs in Docker entrypoints.
- Worker startup belongs in `app/workers` and Docker worker entrypoints.
- Scheduler/beat behavior must be testable without requiring a live exchange.

### 6. Git Workflow

Use focused commits.

```powershell
git status --short
git add app tests alembic scripts pyproject.toml uv.lock
git commit -m "feat: add langgraph trading agent workflow"
```

Rules:

- Do not commit generated caches, coverage output, or local secrets.
- Do not commit experimental scripts unless they are intentionally documented and tested.
- Keep unrelated local changes out of the commit when possible.

---

## Trading, ML, And AI-Agent Rules

### 1. Feature Engineering Consistency

Training, backtesting, diagnostics, and live inference must use shared feature logic.

```python
from app.tools.feature_engineer import calculate_technical_indicators

features = calculate_technical_indicators(market_data)
```

Do not create separate indicator logic for training and inference.

### 2. Strategy, Risk, Guardrails, Execution

The trading decision chain must remain deterministic.

```text
market data
-> model prediction
-> domain.trading_strategy
-> domain.risk_management
-> guardrails.trade_limits / safety_checks
-> services.trade_execution_service
```

Rules:

- Strategy creates a decision.
- Risk management approves, rejects, or skips.
- Guardrails provide final operational allow/block checks.
- Execution service is the only layer that can place market orders.
- API routes and agents must not place exchange orders directly.

### 3. LangGraph Trading Agent

The trading agent is an orchestrator, not the source of trading truth.

Allowed responsibilities:

- Summarize market context.
- Explain model signal.
- Create strategy proposal through domain functions.
- Request human approval when required.
- Run guardrails before execution.
- Call execution service only when live execution is explicitly enabled.

Forbidden responsibilities:

- Bypassing risk management.
- Bypassing guardrails.
- Placing live trades directly through Binance.
- Inventing position sizes outside `risk_management`.
- Treating LLM output as final approval.

### 4. Research Agent And Vector Store

The research agent should use LlamaIndex and `app/vector_store` for read-heavy knowledge workflows.

Allowed use cases:

- Market reports.
- Strategy notes.
- Trading journal search.
- News summaries.
- SEC-style research.
- Historical explanation and retrieval.

Rules:

- Research/RAG output is context, not trade approval.
- Research agent must not execute trades.
- Vector-store data should be versioned or rebuildable.
- Keep retrieval code separate from live execution code.

### 5. Human Approval

Live trading must support human approval gates.

Rules:

- Live execution must be opt-in.
- Risky or high-notional trades must require approval.
- Approval decisions must be recorded in agent result metadata.
- Missing approval should produce `NEEDS_APPROVAL`, not a live order.

---

## Security & Secrets Management

### 1. Environment Variables

Required secrets belong in `.env`, never in code.

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
JWT_SECRET_KEY=change_me
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Rules:

- Commit `.env.example`, never `.env`.
- Do not log tokens, API keys, or passwords.
- Use separate dev, test, and production credentials.
- Prefer read-only exchange keys for data-only workflows.
- Never run live trading with test credentials accidentally.

### 2. AI And External Providers

Rules:

- Do not send secrets to LLM providers.
- Do not send private keys, `.env`, or raw credentials to external AI APIs.
- Archived AI enhancement scripts are reference-only and not active tooling.
- New AI provider integrations must be opt-in, documented, and tested.

---

## Contribution Checklist

Before submitting a PR, verify:

- [ ] Tests pass for changed modules.
- [ ] Ruff passes for changed Python files.
- [ ] MyPy passes for changed typed modules.
- [ ] New app modules have mirrored tests.
- [ ] New database model changes include Alembic migrations.
- [ ] New dependencies are in `pyproject.toml` and `uv.lock`.
- [ ] API routes use schemas and dependency injection.
- [ ] Trading logic stays in domain, guardrails, or services.
- [ ] Agents do not bypass guardrails or execution services.
- [ ] No secrets are committed or logged.
- [ ] README or docs are updated when behavior changes.

---

## Common Mistakes

| Wrong | Correct |
| ----- | ------- |
| Adding new code under `src/` | Add active code under `app/` |
| Route handler contains trading logic | Route calls service/domain layer |
| Agent directly calls Binance | Agent calls guardrails and execution service |
| LLM output approves a trade | Risk and guardrails approve the trade |
| DB table changed without migration | Add an Alembic revision |
| Test hits live Binance/Postgres | Use fake service, mock, or dependency override |
| Installed dependency only in `.venv` | Add it to `pyproject.toml` and `uv.lock` |
| `print()` in production code | Use project logger |
| Broad `except:` | Catch specific exceptions |
| Tests live in old flat layout | Tests mirror `app/` subfolders |

---

## Questions Or Clarifications

If unclear on any standard:

1. Check this document first.
2. Review `README.md` and `CONTRIBUTING.md`.
3. Inspect the matching `tests/` folder for examples.
4. Ask in PR comments or open a focused issue.

---

## Last Updated

**Date**: May 18, 2026

**Version**: 4.0

**Status**: Refactored app architecture with LangGraph trading-agent foundation
