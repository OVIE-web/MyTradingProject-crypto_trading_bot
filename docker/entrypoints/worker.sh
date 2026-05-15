#!/bin/sh
set -e

exec python -m app.workers.scheduler
