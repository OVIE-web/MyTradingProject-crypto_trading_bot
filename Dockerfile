# 1. Use a secure, minimal Python base image
FROM python:3.12.10-slim-bullseye

# 2. Set environment variables for predictable behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# 3. Set working directory
WORKDIR /app

# 4. Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# 5. Install uv for dependency management
RUN pip install "uv==0.4.18"

# 6. Copy dependency metadata first
COPY pyproject.toml ./

# 7. Install dependencies
RUN uv pip install --system .

# 8. Copy project source code
COPY src/ ./src/
COPY main.py .
COPY check_db_connection.py .
COPY tests/ ./tests/  
COPY .env . /

# 9. Add test framework
RUN pip install pytest pytest-asyncio

# 10. Create secure non-root user
RUN adduser --disabled-password --gecos '' appuser

# 11. Ensure models directory exists and is writable by the non-root user
# This prevents runtime errors when the application saves the trained model
RUN mkdir -p /app/src/models && chown -R appuser:appuser /app

# switch to non-root user
USER appuser

# 12. Expose FastAPI, Streamlit, MLflow ports
EXPOSE 8000 8501 5000

# 12. Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# 13. Default: start FastAPI app (after DB check)
CMD ["sh", "-c", "python check_db_connection.py && uvicorn main:app --host 0.0.0.0 --port 8000"]
