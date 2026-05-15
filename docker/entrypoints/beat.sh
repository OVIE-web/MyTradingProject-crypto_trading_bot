#!/bin/sh
set -e

exec celery -A app.workers.celery_worker:celery_app beat --loglevel="${LOG_LEVEL:-INFO}"
