from pathlib import Path

from setuptools import find_packages, setup

# --- Read README for PyPI long description ---
this_dir = Path(__file__).parent
try:
    readme = (this_dir / "README.md").read_text(encoding="utf-8") if (this_dir / "README.md").exists() else ""
except Exception:
    readme = ""

setup(
    name="crypto-trading-bot-1",
    version="1.1.0",
    author="Ovie",
    author_email="oviemsaniyo30@gmail.com",
    description="A modular crypto trading bot with FastAPI backend, PostgreSQL model registry, ML pipeline, and notifier integration.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot-1",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",

    # ------------------------------------------------------------------
    # ✅ Install requirements
    # ------------------------------------------------------------------
    install_requires=[
        # --- Core ---
        "fastapi==0.111.0",
        "uvicorn==0.35.0",
        "sqlalchemy==2.0.41",
        "alembic==1.16.4",
        "psycopg2-binary==2.9.9",
        "python-dotenv==1.1.1",
        "requests==2.32.4",

        # --- Trading + Exchange ---
        "python-binance==1.0.29",
        "python-telegram-bot==22.5",
        "ta==0.11.0",
        "backoff==2.2.1",

        # --- Authentication ---
        "passlib==1.7.4",
        "bcrypt==3.2.2",
        "python-jose==3.3.0",
        "email-validator==2.2.0",
        "python-multipart==0.0.20",

        # --- ML & Data ---
        "pandas==2.3.0",
        "numpy==1.26.4",
        "scikit-learn==1.3.2",
        "xgboost==3.0.2",
        "imbalanced-learn==0.12.4",
        "mlflow==2.13.0",
        "matplotlib==3.10.7",
        "plotly==6.1.2",
        "joblib==1.5.1",

        # --- Utils ---
        "jinja2==3.1.6",
        "pydantic==2.11.7",
    ],

    # ------------------------------------------------------------------
    # ✅ Developer dependencies
    # ------------------------------------------------------------------
    extras_require={
        "dev": [
            "pytest==8.4.1",
            "pytest-asyncio==1.3.0",
            "pytest-mock==3.14.1",
            "flake8==7.3.0",
            "black==24.8.0",
            "isort==5.13.2",
            "mypy==1.13.0",
            "httpx==0.28.1",
            "watchfiles==1.1.0",
            "pip-tools==7.4.1",
            "pandas-stubs==2.3.2.250926",
            "types-setuptools==80.9.0.20250822",
        ],
    },

    # ------------------------------------------------------------------
    # ✅ Classifiers (for PyPI)
    # ------------------------------------------------------------------
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Framework :: FastAPI",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
