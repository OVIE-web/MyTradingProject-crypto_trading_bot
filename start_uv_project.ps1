<#
============================================
 start_uv_project.ps1
 Initialize and activate uv-managed Python environment
 for the Crypto Trading Bot project (Windows PowerShell)
============================================
#>

# Stop on error
$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting project setup with uv..."

# Navigate to project root
Set-Location -Path $PSScriptRoot

# Check for uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "⚙️ uv not found — installing via pip..."
    pip install --upgrade uv
}

# Create venv if missing
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "🧱 Creating virtual environment with uv..."
    uv venv
} else {
    Write-Host "✅ Virtual environment already exists."
}

# Activate venv
Write-Host "🔌 Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "📦 Installing dependencies..."
    uv pip install -r requirements.txt
}

# Install editable project from pyproject.toml
if (Test-Path "pyproject.toml") {
    Write-Host "🔧 Installing editable project package..."
    uv pip install -e .
}

# Install dev dependencies
if (Test-Path "requirements-dev.txt") {
    Write-Host "🧪 Installing development dependencies..."
    uv pip install -r requirements-dev.txt
}

# ------------------------------------------------------------
# ✅ POST-SETUP HEALTH CHECK
# ------------------------------------------------------------
Write-Host "`n🔍 Running post-setup health checks..."

# 1️⃣ Python interpreter check
Write-Host "🐍 Checking Python environment..."
python --version

# 2️⃣ Core imports verification
Write-Host "🧠 Checking core library imports..."
python - <<'PYCODE'
import sys
import fastapi, sqlalchemy, pandas, numpy, requests
print("✅ Core libraries imported successfully!")
PYCODE

# 3️⃣ .env presence check
if (Test-Path ".env") {
    Write-Host "🔑 .env file found."
} else {
    Write-Host "⚠️ No .env file found — please create or copy from .env.example"
}

# 4️⃣ PostgreSQL connectivity test (optional)
Write-Host "🗄️ Checking database connectivity (if configured)..."
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

# 5️⃣ FastAPI app import test
Write-Host "🌐 Validating FastAPI app import..."
python - <<'PYCODE'
try:
    from src.main_api import app
    print("✅ FastAPI app imported successfully.")
except Exception as e:
    print(f"❌ FastAPI app failed to import: {e}")
    sys.exit(1)
PYCODE

# 6️⃣ Directory structure verification
$dirs = @("src", "tests", "data")
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "📁 Directory '$dir' found."
    } else {
        Write-Host "⚠️ Missing directory: $dir"
    }
}

# 7️⃣ Launch VS Code (optional)
if (Get-Command code -ErrorAction SilentlyContinue) {
    Write-Host "🧠 Launching VS Code..."
    code .
}

Write-Host "`n✅ All setup checks completed successfully!"
Write-Host "💡 You can now run: uv run uvicorn src.main_api:app --reload"