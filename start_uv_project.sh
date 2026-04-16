#!/usr/bin/env bash
# ============================================
# start_uv_project.sh
# Initialize and activate uv-managed Python environment
# for the Crypto Trading Bot project (Linux/macOS)
# ============================================

set -e  # Exit immediately on error

# Navigate to project root
cd "$(dirname "$0")"

echo "🚀 Starting project setup with uv..."

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "⚙️ uv not found — installing with pip..."
    pip install --upgrade uv
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "🧱 Creating virtual environment with uv..."
    uv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate venv
echo "🔌 Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Install main dependencies
echo "📦 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
fi

# Install editable project
if [ -f "pyproject.toml" ]; then
    echo "🔧 Installing editable project package..."
    uv pip install -e .
fi

# Install dev dependencies
if [ -f "requirements-dev.txt" ]; then
    echo "🧪 Installing development dependencies..."
    uv pip install -r requirements-dev.txt
fi

# ------------------------------------------------------------
# ✅ POST-SETUP HEALTH CHECK
# ------------------------------------------------------------
echo
echo "🔍 Running post-setup health checks..."

# 1️⃣ Check Python interpreter
python_version=$(python -V)
echo "🐍 Python environment active: $python_version"

# 2️⃣ Check that FastAPI and key libs import successfully
echo "🧠 Checking core imports..."
python - <<'PYCODE'
import sys
import fastapi, sqlalchemy, pandas, numpy, requests
print("✅ Core libraries imported successfully!")
PYCODE

# 3️⃣ Check if .env and config are readable
if [ -f ".env" ]; then
    echo "🔑 .env file found."
else
    echo "⚠️ No .env file found — you may need to copy from .env.example"
fi

# 4️⃣ Optional PostgreSQL connectivity check
echo "🗄️ Checking database connectivity (if configured)..."
python - <<'PYCODE'
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

db_url = os.getenv("DATABASE_URL") or \
          f"postgresql://{os.getenv('POSTGRES_USER','user')}:{os.getenv('POSTGRES_PASSWORD','pass')}@{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','postgres')}"

try:
    conn = psycopg2.connect(db_url)
    print("✅ PostgreSQL connection successful.")
    conn.close()
except Exception as e:
    print(f"⚠️ Could not connect to PostgreSQL: {e}")
PYCODE

# 5️⃣ Check FastAPI app import
echo "🌐 Validating FastAPI app import..."
python - <<'PYCODE'
try:
    from src.main_api import app
    print("✅ FastAPI app imported successfully.")
except Exception as e:
    print(f"❌ FastAPI app failed to import: {e}")
    sys.exit(1)
PYCODE

# 6️⃣ Directory structure check
for dir in "src" "tests" "data"; do
    if [ -d "$dir" ]; then
        echo "📁 Directory '$dir' found."
    else
        echo "⚠️ Missing directory: $dir"
    fi
done

# Launch VS Code if installed
if command -v code &> /dev/null; then
    echo "🧠 Launching VS Code..."
    code .
fi

$password = $env:ADMIN_PASSWORD
$bytes = [System.Text.Encoding]::UTF8.GetByteCount($password)
Write-Host "Password length: $bytes bytes"
if ($bytes -gt 72)
    Write-Host "ERROR: Password exceeds 72 bytes!"
    Write-Host "Use: $($password.Substring(0, 72))"
fi

echo
echo "✅ All setup checks completed successfully!"
echo "💡 You can now run: uv run uvicorn src.main_api:app --reload"
