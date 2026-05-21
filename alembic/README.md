# Alembic Migrations

This folder contains versioned database migrations for the AI-powered crypto trading bot system.

Common commands:

```powershell
python -m alembic current
python -m alembic upgrade head
python -m alembic revision --autogenerate -m "describe change"
```

Alembic reads the database URL from `app.core.config.DATABASE_URL`, so keep `.env` aligned with the database you want to migrate.
