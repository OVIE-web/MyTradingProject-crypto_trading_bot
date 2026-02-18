from pathlib import Path

from setuptools import find_packages, setup

# --- Read README for PyPI long description ---
this_dir = Path(__file__).parent
try:
    readme = (
        (this_dir / "README.md").read_text(encoding="utf-8")
        if (this_dir / "README.md").exists()
        else ""
    )
except Exception:
    readme = ""

# --- Core dependencies ---
install_requires: list[str] = [
    # --- Web Framework ---
    "fastapi==0.111.0",
    "uvicorn==0.35.0",
    "python-multipart==0.0.20",
    # --- Database ---
    "sqlalchemy==2.0.41",
    "alembic==1.16.4",
    "psycopg2-binary==2.9.9",
    # --- API & HTTP ---
    "requests==2.32.4",
    "httpx==0.28.1",
    # --- Configuration ---
    "python-dotenv==1.1.1",
    "pydantic==2.11.7",
    # --- Trading & Exchange ---
    "python-binance==1.0.29",
    "ta==0.11.0",
    "backoff==2.2.1",
    # --- Notifications ---
    "python-telegram-bot==22.5",
    # --- Authentication ---
    "passlib==1.7.4",
    "bcrypt==3.2.2",
    "python-jose==3.3.0",
    "email-validator==2.2.0",
    # --- Data Science & ML ---
    "pandas==2.3.0",
    "numpy==1.26.4",
    "scikit-learn==1.3.2",
    "xgboost==3.0.2",
    "imbalanced-learn==0.12.4",
    "joblib==1.5.1",
    # --- Visualization & Dashboard ---
    "matplotlib==3.10.7",
    "plotly==6.1.2",
    "streamlit==1.53.1",
    # --- Experiment Tracking ---
    "mlflow==2.13.0",
    # --- Templates ---
    "jinja2==3.1.6",
]

# --- Developer dependencies ---
extras_require: dict[str, list[str]] = {
    "dev": [
        "pytest==8.4.1",
        "pytest-asyncio==1.3.0",
        "pytest-mock==3.14.1",
        "pytest-cov==6.0.0",
        "mypy==1.13.0",
        "ruff==0.8.4",
        "watchfiles==1.1.0",
        "pip-tools==7.4.1",
        "types-requests==2.33.0.20250130",
        "types-psycopg2==2.9.21.15",
        "types-python-jose==3.3.8.20240106",
        "types-passlib==1.7.7.20240106",
        "pandas-stubs==2.3.2.250926",
    ],
}

setup(
    name="crypto-trading-bot-1",
    version="1.1.0",
    author="Ovie",
    author_email="oviemsaniyo30@gmail.com",
    description="A modular crypto trading bot with FastAPI backend, PostgreSQL model registry, ML pipeline, and notifier integration.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,  # type: ignore[arg-type]
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: FastAPI",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "crypto",
        "trading",
        "bot",
        "xgboost",
        "machine-learning",
        "binance",
        "fastapi",
    ],
)
