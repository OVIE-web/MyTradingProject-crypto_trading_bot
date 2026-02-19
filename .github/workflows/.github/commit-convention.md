# 🧩 Commit Convention — Crypto Trading Bot Project

This document defines the **Conventional Commit** rules and message style for all commits, pull requests, and automated changes made by humans or GitHub Copilot.

---

## 🧠 Purpose

Consistent commit messages help the team:

- Maintain a **clear, searchable project history**
- Enable **semantic versioning** and **changelog automation**
- Improve **CI/CD pipelines** (releases, deployments, version tags)
- Enhance **readability and traceability** across commits

---

## 🧱 Commit Message Format

Each commit message must follow this structure:

type (optional scope): <short, imperative description>

[optional body]

[optional footer]

## Example

feat(notifier): add async retry logic for Telegram notifications

- This commit introduces asynchronous retry logic to the Telegram notification system, improving reliability in message delivery.

---

## 🧩 Allowed Commit Types

| Type | Description |
| ------ | -------------- |
| **feat:** | Introduces a new feature (code, config, endpoint, etc.) |
| **fix:** | Fixes a bug or incorrect behavior |
| **refactor:** | Refactors code without changing behavior |
| **test:** | Adds or updates test cases |
| **docs:** | Documentation-only updates (README, wiki, docstrings) |
| **style:** | Linting, formatting, or stylistic code changes |
| **cicd:** | CI/CD workflow, Docker, or deployment configuration updates |
| **chore:** | Maintenance, dependency upgrades, or cleanup |
| **perf:** | Performance optimization or improvement |
| **build:** | Build-related changes (Dockerfile, setup.py, pyproject.toml) |

✅ **Use lowercase** for all commit types and scopes.

---

## 🧩 Scopes

Scopes identify which part of the project was affected.

| Scope | Description |
| -------- | ------------- |
| **api** | FastAPI routes, endpoints, and controllers |
| **db** | Database logic, models, and migrations |
| **ml** | Machine learning modules and model training |
| **notifier** | Telegram, Email, or notification system |
| **tests** | Unit, integration, or mock tests |
| **config** | Application configuration and environment settings |
| **ci/cd** | GitHub Actions, Docker, or pipeline automation |
| **docs** | Documentation, wiki, or README updates |
| **infra** | Infrastructure, Docker, or deployment setup |

### Example with scope

fix(db): resolve connection pooling issue in SQLAlchemy

- This commit fixes a connection pooling issue in the database layer using SQLAlchemy, preventing potential timeouts during high traffic.

---

## 🧩 Commit Body (Optional)

Use the **body** to explain *why* the change was made, not just *what* changed.

- Separate summary and body with a blank line.
- Use **present tense**, **imperative mood** (“add” not “added”).
- Wrap lines at ~72 characters.

Example:

feat(model_manager): add SHAP feature importance visualization

- This commit introduces SHAP-based feature importance visualization to the model manager module. This helps in understanding model predictions better and aids in debugging and improving model performance.

---

## 🧩 Footer (Optional)

Use footers for:

- Issue references (`Fixes #45`, `Closes #99`)
- Breaking changes

Example:

BREAKING CHANGE: renamed field signal_confidence to confidence_score

- This change affects all modules that interact with the signal confidence metric. Update your code accordingly.

---

## 🔒 Best Practices

- Keep commits **atomic** — one purpose per commit.  
- Always **run tests and linting** before committing:

```bash
  uv run pytest -v
  uv run ruff src/ test/
  uv run mypy src/ tests/

Never commit secrets, credentials, or .env files.

Reference issues or pull requests when applicable.

Keep messages short (max ~100 characters for the title).

Use the body for detailed explanations when necessary.

---

## 🧪 Example Commits

feat(api): add /predict endpoint for model inference
fix(notifier): handle missing chat_id gracefully in TelegramNotifier
test(ml): mock Binance API in model_manager tests
refactor(config): simplify environment variable loading
docs: update installation guide in README
ci/cd: add linting and pytest jobs to GitHub Actions
chore: bump FastAPI and XGBoost versions
build: update Dockerfile for uv-based dependency management
perf(model_manager): optimize data preprocessing pipeline

---

##  🧰 Code Review Checklist

Before merging, confirm that:

✅ Code passes linting (ruff, mypy)

✅ Tests pass locally and in CI/CD (pytest -v)

✅ Commit message follows this convention

✅ No secrets or API keys are included

✅ Functions have docstrings and type hints

✅ Dependencies are pinned in pyproject.toml

✅ Dockerfile is up-to-date

✅ GitHub Actions workflow is up-to-date

✅ CI/CD pipeline is up-to-date

🧠 Summary

✅ Use Conventional Commits

✅ Follow scoped, clear commit types

✅ Keep commits atomic and documented

✅ Ensure tests and lint checks pass before pushing

✅ Reference issues and PRs when relevant

“Every commit is a story — make it worth reading.”

Author: Ovie Saniyo
Project: 🧠 Crypto Trading Bot 1
License: MIT
Last Updated: February 2026

👋 See you next time!
