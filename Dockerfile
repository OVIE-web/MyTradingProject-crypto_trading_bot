# =====================================================
# 🧱 Stage 1: Base Environment
# =====================================================
FROM python:3.12.10-slim-bullseye AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Install core system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN pip install "uv==0.4.18"

# Copy dependency metadata first (for optimal Docker layer caching)
COPY pyproject.toml uv.lock ./

# Copy source code (required by setuptools for installation)
COPY src/ ./src/

# Install Python dependencies (system-wide, cached layer)
RUN uv pip install --system .

# Sync development dependencies into virtual environment
RUN uv sync --frozen

# Add non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Create necessary writable directories with proper permissions
RUN mkdir -p /app/src/models /app/logs && chown -R appuser:appuser /app

USER appuser


# =====================================================
# 🧠 Stage 2: Final Runtime Image
# =====================================================
FROM base AS runtime

WORKDIR /app

# Files are already copied in base stage, no need to copy again
# But we can copy main.py if it exists separately from src/
COPY --chown=appuser:appuser main.py .

EXPOSE 8000 8501 5000

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# =====================================================
# 🧪 Stage 3: API Server
# =====================================================
FROM runtime AS api

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# =====================================================
# 🤖 Stage 4: Trading Bot
# =====================================================
FROM runtime AS bot

WORKDIR /app

CMD ["python", "-m", "src.bot.main"]


# =====================================================
# 🧪 Stage 5: Test Runner
# =====================================================
FROM base AS test

WORKDIR /app

USER root

# Install additional testing tools
RUN pip install pytest==8.4.1 pytest-asyncio==1.3.0 pytest-mock==3.14.1

USER appuser

CMD ["pytest", "-v", "--maxfail=3", "--disable-warnings", "--tb=short", "src/tests/"]