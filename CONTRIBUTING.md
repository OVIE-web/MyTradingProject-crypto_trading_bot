# Contributing Guidelines

Thank you for considering contributing to the Crypto Trading Bot project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [oviemsaniyo30@gmail.com](mailto:oviemsaniyo30@gmail.com).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker and Docker Compose (for local testing)
- PostgreSQL 13+ (or use Docker)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot.git
cd MyTradingProject-crypto_trading_bot

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install UV package manager
pip install uv

# Install dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Local Development with Docker

```bash
# Start services
docker-compose up -d

# Run tests
docker-compose run --rm test

# View logs
docker-compose logs -f
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b bugfix/your-bug-fix
```

### 2. Make Your Changes

- Write clean, well-documented code
- Follow PEP 8 and project conventions
- Add type hints to all functions
- Write or update tests for your changes

### 3. Run Tests Locally

```bash
# Run all tests
uv run python -m pytest -v

# Run specific test file
uv run python -m pytest tests/test_settings.py -v

# Run with coverage
uv run python -m pytest --cov=src --cov-report=html

# Format code
uv run ruff format src/ tests/

# Check for issues
uv run ruff check src/ tests/

# Type checking
uv run python -m mypy src/
```

### 4. Commit Your Changes

See [Commit Message Guidelines](#commit-message-guidelines) below.

```bash
git add .
git commit -m "feat: add new trading feature"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub (see [Pull Request Process](#pull-request-process)).

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use f-strings for string formatting

### Type Hints Example

```python
from typing import Optional, List
from pathlib import Path

def process_data(
    file_path: Path,
    threshold: float = 0.5,
    tags: Optional[List[str]] = None,
) -> dict[str, any]:
    """Process data file and return results.
    
    Args:
        file_path: Path to input CSV file
        threshold: Minimum confidence threshold
        tags: Optional list of tags to filter
        
    Returns:
        Dictionary containing processed results
    """
    pass
```

### Docstring Format

Use Google-style docstrings:

```python
def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI) for price data.
    
    Args:
        prices: List of closing prices
        period: RSI calculation period (default: 14)
        
    Returns:
        RSI value between 0 and 100
        
    Raises:
        ValueError: If prices list is too short for the given period
        
    Example:
        >>> prices = [100, 101, 102, 101, 100]
        >>> rsi = calculate_rsi(prices, period=3)
        >>> 0 <= rsi <= 100
        True
    """
    pass
```

## Testing Requirements

### Minimum Coverage

- Aim for **70%+ code coverage**
- Critical paths should have **90%+ coverage**
- All public APIs must be tested

### Test File Naming

```
tests/
├── test_settings.py           # Unit tests
├── notifications/
│   ├── test_notification.py
│   └── test_notifier.py
└── integration/
    └── test_database.py       # Integration tests
```

### Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock

class TestFeatureName:
    """Test suite for feature_name module."""
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample data for tests."""
        return {"key": "value"}
    
    def test_happy_path(self, sample_data):
        """Test successful operation."""
        result = function_under_test(sample_data)
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run specific test
uv run python -m pytest tests/test_settings.py::test_get_env_missing_required

# Run with coverage report
uv run python -m pytest --cov=src --cov-report=html --cov-report=term

# Run tests matching pattern
uv run python -m pytest -k "test_trading" -v
```

## Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, semicolons, etc.)
- **refactor**: Code refactoring without feature changes
- **perf**: Performance improvements
- **test**: Test additions or modifications
- **ci**: CI/CD configuration changes
- **chore**: Build system, dependencies, or tooling changes

### Examples

```bash
# Feature
git commit -m "feat(trading): add RSI indicator calculation

- Implement RSI calculation with configurable period
- Add unit tests for edge cases
- Update documentation with usage examples"

# Bug fix
git commit -m "fix(db): resolve connection timeout issues

- Increase connection timeout threshold
- Add retry logic with exponential backoff
- Fixes #123"

# Documentation
git commit -m "docs: update README with GCP deployment guide"

# Test
git commit -m "test: add integration tests for Binance API"
```

## Pull Request Process

### Before Submitting

1. **Update your branch** with latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run full test suite**:
   ```bash
   uv run python -m pytest -v
   uv run ruff check src/ tests/
   uv run python -m mypy src/
   ```

3. **Update documentation** if needed

4. **Ensure all tests pass** in GitHub Actions

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issues
Fixes #123

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests passing locally

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. At least one maintainer review required
2. All conversations must be resolved
3. All checks (tests, linting, type checking) must pass
4. CI/CD pipeline must be green

## Reporting Issues

### Before Reporting

- Check existing issues to avoid duplicates
- Update to the latest version
- Search closed issues for similar problems

### Issue Report Format

Include:

1. **Title**: Clear, descriptive summary
2. **Environment**:
   ```
   - Python version
   - OS
   - Docker version (if applicable)
   ```
3. **Reproduction Steps**: Step-by-step to reproduce
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Error Messages**: Full stack traces
7. **Logs**: Relevant log output
8. **Screenshots**: If applicable

## Documentation

### Code Documentation

- Docstrings required for all public functions/classes
- Include type hints in all signatures
- Provide usage examples in docstrings
- Update docstrings if behavior changes

### Project Documentation

- Update README.md for user-facing changes
- Update CONTRIBUTING.md for process changes
- Add examples for new features
- Keep API documentation current

### Documentation Tools

```bash
# Generate documentation locally
pip install sphinx
sphinx-build -b html docs/ docs/_build/

# Preview: open docs/_build/index.html
```

## Questions?

- Check [GitHub Discussions](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/discussions)
- Open an [issue](https://github.com/OVIE-web/MyTradingProject-crypto_trading_bot/issues)
- Email: [oviemsaniyo30@gmail.com](mailto:oviemsaniyo30@gmail.com)

## Recognition

Contributors will be recognized in:
- Release notes
- Contributors list in README
- Commit history

Thank you for contributing! 🎉