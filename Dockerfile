FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY docker ./docker
COPY pyproject.toml ./

RUN mkdir -p /app/logs /app/models \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser


FROM base AS api

EXPOSE 8000
CMD ["sh", "docker/entrypoints/api.sh"]


FROM base AS worker

CMD ["sh", "docker/entrypoints/worker.sh"]


FROM base AS celery-worker

CMD ["celery", "-A", "app.workers.celery_worker:celery_app", "worker", "--loglevel=info"]


FROM base AS celery-beat

CMD ["sh", "docker/entrypoints/beat.sh"]


FROM base AS streamlit

EXPOSE 8501
CMD ["streamlit", "run", "app/frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]


FROM base AS test

COPY requirements.dev.txt ./
RUN pip install -r requirements.dev.txt

COPY tests ./tests
CMD ["pytest", "-q"]
