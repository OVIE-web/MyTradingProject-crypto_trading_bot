# PowerShell script using uv to create and manage the Python environment

# Navigate to the script directory (your project root)
Set-Location -Path $PSScriptRoot

# Check if .venv exists
if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment with uv..."
    uv venv
}

# Activate the virtual environment
Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

# Install dependencies from pyproject.toml
Write-Host "Installing dependencies with uv..."
uv pip install -r requirements.txt
# If you have a pyproject.toml, you can use:
uv pip install -e .
# Check if uv is installed
if (-Not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing uv..."
    pip install uv
}

# Launch VS Code
Write-Host "Launching VS Code..."
code .
