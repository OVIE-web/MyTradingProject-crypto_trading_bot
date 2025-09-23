# 1. Use a secure, minimal Python image
FROM python:3.12.10-slim-bullseye

# 2. Set environment variables for consistent behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 3. Set work directory inside the container
WORKDIR /app

# 4. Install uv for fast dependency resolution and install tools
RUN pip install uv

# 5. Create virtual environment
RUN uv venv

# 6. Copy Linux-specific project metadata first
COPY pyproject.linux.toml ./pyproject.toml

# 7. Install dependencies (from pyproject.toml)
RUN . .venv/bin/activate && uv pip install .

# 8. Copy project source code
COPY src/ ./src/
COPY main.py .

# 9. Expose the commonly used ports
EXPOSE 8000 8501 5000

# 10. Use FastAPI's production server (uvicorn) as default CMD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]