# =====================================================
# 🧱 Stage 1: Base Environment
# =====================================================
FROM python:3.12.10-slim-bullseye AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    UV_HTTP_TIMEOUT=120

WORKDIR /app

# Install core system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN pip install "uv==0.4.18"

# Copy dependency metadata first (for caching)
COPY pyproject.toml ./

# Install Python dependencies system-wide
RUN uv pip install --system .

# Add non-root user for security
RUN adduser --disabled-password --gecos '' appuser

# Create necessary writable directories
RUN mkdir -p /app/src/models /app/logs && chown -R appuser:appuser /app
USER appuser


# =====================================================
# 🧠 Stage 2: Final Runtime Image
# =====================================================
FROM base AS runtime

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser main.py .

EXPOSE 8000 8501 5000

# Default: run Streamlit dashboard
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# =====================================================
# 🧪 Stage 3: Test Runner (optional)
# =====================================================
FROM runtime AS test
USER root
RUN pip install pytest pytest-asyncio pytest-mock
USER appuser
CMD ["pytest", "-v", "--maxfail=1", "--disable-warnings"]
