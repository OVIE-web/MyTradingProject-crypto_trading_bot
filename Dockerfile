# 1. Use a secure, minimal Python image
FROM python:3.12.10-slim-bullseye

# 2. Set environment variables for consistent behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3. Set work directory inside the container
WORKDIR /app

# 4. Install uv for fast dependency resolution
RUN pip install "uv==0.4.18"

# 5. Copy project metadata first (for dependency caching)
COPY pyproject.toml ./

# 6. Install dependencies into system environment
RUN uv pip install --system .

# 7. Copy project source code
COPY src/ ./src/
COPY main.py .

# 8. Add non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# 9. Expose the commonly used ports
EXPOSE 8000 8501 5000

# 10. Healthcheck (optional)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# 11. Use FastAPI's production server (uvicorn) as default CMD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]