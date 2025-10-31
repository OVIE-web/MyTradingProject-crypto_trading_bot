# Rules for AI Interaction

## Project Structure
- All historical data files must be placed in the `data/` directory
- Model files should be saved to the configured model directory (controlled by MODEL_DIR env var)
- Tests must be placed in the `tests/` directory following the pattern `test_*.py`
- Source code belongs in the `src/` directory

## Code Standards
1. All Python code must:
   - Be compatible with Python 3.12+
   - Follow PEP 8 style guidelines
   - Include type hints for function parameters and return values
   - Include docstrings for modules, classes, and functions

2. Testing Requirements:
   - All new features must include corresponding tests
   - Tests should use pytest fixtures where appropriate
   - Mock external services (Binance API, database) in tests
   - Maintain test coverage above 80%

3. Documentation:
   - Keep README.md up to date with new features
   - Document all environment variables in .env.example
   - Include clear error messages and logging
   - Add comments for complex algorithms or business logic

## Development Workflow
1. Environment Setup:
   - Use uv for dependency management
   - Always update requirements.txt when adding new dependencies
   - Keep pyproject.toml in sync with requirements

2. Docker:
   - Update docker-compose.yml when adding new services
   - Use volumes for persistent data (models, databases)
   - Document any new environment variables needed for containers

3. Database:
   - Follow database migration best practices
   - Update pg_hba.conf when changing authentication rules
   - Include database setup scripts in scripts/

## Security
1. Never commit:
   - .env files with real credentials
   - API keys or secrets
   - Personal trading data
   - Large model files or datasets

2. Always use:
   - Environment variables for configuration
   - Secure connection strings
   - Rate limiting for API calls
   - Input validation for all data

## Trading Bot Specific
1. Risk Management:
   - Implement position size limits
   - Include stop-loss mechanisms
   - Add logging for all trades
   - Validate order parameters

2. Model Management:
   - Version control trained models
   - Document feature engineering steps
   - Include model performance metrics
   - Save model evaluation results

3. Data Processing:
   - Validate data quality before training
   - Handle missing values appropriately
   - Document data preprocessing steps
   - Include data sanity checks

## Contribution Guidelines
1. Code Changes:
   - Write clear commit messages
   - Update tests for new features
   - Follow existing code style
   - Document breaking changes

2. Review Process:
   - Run full test suite before commits
   - Update documentation as needed
   - Check for security implications
   - Verify Docker builds successfully
