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

# 5. Copy only project metadata first (improves Docker layer caching)
COPY pyproject.toml ./

# 6. Install dependencies (from pyproject.toml)
RUN uv pip install -e . && uv pip install .[dev]

# 7. Copy project source code
COPY src/ ./src/
COPY main.py .

# 8. Copy any saved models (XGBoost or others)
COPY src/models/ ./src/models/

# 9. Expose the commonly used ports
EXPOSE 8000 8501 5000

# 10. Use FastAPI's production server (uvicorn) as default CMD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# 11. Use a non-root user for security (optional, but recommended)
# RUN adduser --disabled-password --gecos '' appuser