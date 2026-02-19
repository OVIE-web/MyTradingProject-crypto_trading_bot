# 🤖 Copilot Custom Instructions – Crypto Trading Bot Project

These concise rules guide Copilot’s behavior when generating, refactoring, or reviewing code for this project.

---

## 🧠 General Guidelines

- Always produce **production-ready Python 3.12 code** that follows **PEP 8** and uses **type hints + docstrings**.
- Keep code **modular**, **async-safe**, and **secure** — no hardcoded credentials, only `os.getenv()`.
- Structure logic around the project’s architecture (`src/`, `tests/`, `notifications/`, etc.).
- Use **logging**, not `print()`.
- Follow the configuration rules in `pyproject.toml`.

---

## 🧩 Code Generation Rules

### FastAPI

- Use `async def` endpoints with Pydantic validation.
- Return structured responses with proper status codes.
- Include at least one test for each endpoint.

### Database (SQLAlchemy)

- Use `DeclarativeBase` models and `SessionLocal()` for sessions.
- Wrap DB operations in try/except blocks.
- Ensure migrations are consistent with Alembic.

### Machine Learning

- Use `pandas`, `numpy`, `scikit-learn`, `xgboost`.
- Save models under `/models` with clear versioning.
- Implement reproducible results with random seeds.

### Notifications

- Use async APIs (`python-telegram-bot`, `aiosmtplib`).
- Apply retry logic via `backoff`.
- Centralize logic in `notifier.py`.

### Formatting & Linting

- Auto-format with **ruff** before commits.
- Match rules in `pyproject.toml`.

---

## 🧪 Testing Standards

- Use `pytest` + `pytest-asyncio`.
- Mirror `src/` structure inside `tests/`.
- Mock external APIs (Telegram, Binance).
- Include Arrange–Act–Assert sections.
- Avoid real network or DB calls unless mocked.

---

## 🧰 Commit Message Style (Conventional Commits)

| Type        | Example                                          |
| ----------- | ------------------------------------------------ |
| `feat:`     | `feat: add Telegram retry logic to notifier.py`  |
| `fix:`      | `fix: correct DB connection issue in config.py`  |
| `test:`     | `test: mock Binance API for model_manager tests` |
| `ci/cd:`    | `ci: optimize GitHub Actions with uv sync`       |
| `refactor:` | `refactor: simplify feature_engineer data prep`  |

Keep messages imperative and under 72 characters.

---

## 🧩 Code Review & CI/CD

All commits must pass `pytest` and `mypy` (static type checking).

```bash
uv pip compile pyproject.toml -o requirements.txt
uv pip compile --extra dev pyproject.toml -o requirements.dev.txt

## 🔒 Security Guidelines

Auto-format with **ruff** before commits.
Match rules in `pyproject.toml`.

### Static Type Checking
- Run `mypy` for static type checking; ensure no type errors before commit.
Validate all input to prevent injection.

Review dependencies quarterly (uv pip list --outdated).

Keep .env files out of version control.

## 🧠 Copilot Behavior Summary

Copilot should:

Write clean, modular, and async-friendly Python.

Follow CI, testing, and security best practices.

Generate readable, documented, and test-covered code.

Always prioritize clarity, reliability, and maintainability.

Author: Ovie Saniyo
Project: 🧠 Crypto Trading Bot 1
Status: Active Development
Last Updated: February 2026
```
