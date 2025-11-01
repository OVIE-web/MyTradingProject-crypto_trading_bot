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
ENV UV_HTTP_TIMEOUT=120
RUN uv pip install --system .

# 8. Copy project source code

# Copy only necessary source files
COPY src/ ./src/
COPY main.py .
# Do not copy check_db_connection.py or tests/ unless needed
# COPY check_db_connection.py .
# COPY tests/ ./tests/
# Only copy .env if you want to bake secrets into the image (not recommended for production)
# COPY .env . /


# 9. (Optional) Add test framework if you plan to run tests in the container
# RUN pip install pytest pytest-asyncio

# 10. Create secure non-root user
RUN adduser --disabled-password --gecos '' appuser

# 11. Ensure models directory exists and is writable by the non-root user
# This prevents runtime errors when the application saves the trained model
RUN mkdir -p /app/src/models && chown -R appuser:appuser /app

# switch to non-root user
USER appuser

# 12. Expose FastAPI, Streamlit, MLflow ports
EXPOSE 8000 8501 5000


# 12. Healthcheck (optional, adjust for your main service)
# HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1


# 13. Default: start Streamlit app (adjust as needed)
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
