#!/usr/bin/env python3
"""
Enhance trading bot using Google Gemini API (google.genai package)
"""

import logging
import os
import sys
from pathlib import Path

import google.genai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure API
api_key: str | None = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    logger.error("Set GOOGLE_API_KEY environment variable")
    logger.error("Run: $env:GOOGLE_API_KEY = 'your-key'")
    sys.exit(1)

logger.info(f"API Key found: {api_key[:10]}...")

client = genai.Client(api_key=api_key)


def read_file(filepath: str) -> str:
    """Read file contents"""
    with open(filepath) as f:
        return f.read()


def save_file(filepath: str, content: str) -> None:
    """Save content to file"""
    # Strip markdown code block wrappers if present
    content = content.strip()
    if content.startswith("```python"):
        content = content[9:].lstrip("\n")
    if content.startswith("```"):
        content = content[3:].lstrip("\n")
    if content.endswith("```"):
        content = content[:-3].rstrip("\n")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def generate_tests(file_path: str) -> str | None:
    """Generate tests for a Python file"""
    if not Path(file_path).exists():
        logger.info(f"Skipping {file_path} (not found)")
        return ""

    logger.info(f"Generating tests for {file_path}...")

    code = read_file(file_path)

    prompt = f"""Analyze this Python code and generate comprehensive test cases using pytest.
Include:
- Normal test cases
- Edge cases
- Error handling
- Mock objects for external dependencies

Code:
```python
{code}
```

Provide ONLY the test code, no explanations. Start with imports and end with test functions."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        test_code = response.text

        # Save tests
        test_file = f"tests/test_{Path(file_path).stem}.py"
        if test_code is not None:
            save_file(test_file, test_code)

        logger.info(f"Tests saved to {test_file}")
        return test_code
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""


def add_documentation(file_path: str) -> str | None:
    """Add Google-style docstrings"""
    if not Path(file_path).exists():
        logger.info(f"Skipping {file_path} (not found)")
        return None

    logger.info(f"Adding documentation to {file_path}...")

    code = read_file(file_path)

    prompt = f"""Add comprehensive Google-style docstrings to all functions and classes.
Also add type hints to all function parameters and returns.

Code:
```python
{code}
```

Provide ONLY the updated code, no explanations."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        documented_code = response.text

        # Save documented code
        output_file = file_path.replace(".py", "_documented.py")
        if documented_code is not None:
            save_file(output_file, documented_code if documented_code is not None else "")
            logger.info(f"Documentation added to {output_file}")
            return documented_code
        else:
            logger.error("No documented code generated.")
            return ""
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""


def refactor_code(file_path: str) -> str | None:
    """Refactor code for better quality"""
    if not Path(file_path).exists():
        logger.info(f"Skipping {file_path} (not found)")
        return None

    logger.info(f"Refactoring {file_path}...")

    code = read_file(file_path)

    prompt = f"""Refactor this Python code to improve:
- async patterns
- error handling
- testability
- code structure

Code:
```python
{code}
```

Provide ONLY the refactored code, no explanations."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        refactored_code = response.text

        # Save refactored code
        output_file = file_path.replace(".py", "_refactored.py")
        if refactored_code is not None:
            save_file(output_file, refactored_code)

        logger.info(f"Refactored code saved to {output_file}")
        return refactored_code
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def optimize_performance(file_path: str) -> str | None:
    """Optimize code for performance"""
    if not Path(file_path).exists():
        logger.info(f"Skipping {file_path} (not found)")
        return None

    logger.info(f"Optimizing {file_path}...")

    code = read_file(file_path)

    prompt = f"""Optimize this Python code for performance. Focus on:
- Database query optimization
- Memory usage
- Caching opportunities
- Connection pooling

Code:
```python
{code}
```

Provide ONLY the optimized code, no explanations."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        optimized_code = response.text

        # Save optimized code
        output_file = file_path.replace(".py", "_optimized.py")
        if optimized_code is not None:
            save_file(output_file, optimized_code)

        logger.info(f"Optimized code saved to {output_file}")
        return optimized_code
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def add_type_hints(file_path: str) -> str | None:
    """Add type hints to all functions"""
    if not Path(file_path).exists():
        logger.info(f"Skipping {file_path} (not found)")
        return None

    logger.info(f"Adding type hints to {file_path}...")

    code = read_file(file_path)

    prompt = f"""Add complete type hints to all functions and variables.
Make the code mypy compliant.

Code:
```python
{code}
```

Provide ONLY the code with type hints, no explanations."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        typed_code = response.text

        # Save typed code
        output_file = file_path.replace(".py", "_typed.py")
        if typed_code is not None:
            save_file(output_file, typed_code)

        logger.info(f"Type hints added to {output_file}")
        return typed_code
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def main() -> None:
    """Main enhancement routine"""
    logger.info("Trading Bot Enhancement with Google Gemini (NEW API)")
    logger.info("=" * 60)

    # Files to enhance
    files_to_enhance = [
        "src/model_registry.py",
        "src/visualizer.py",
        "src/train_model.py",
        "src/bot_runner.py",
        "src/db.py",
    ]

    # Check which files exist
    files_to_process = []
    for f in files_to_enhance:
        if Path(f).exists():
            files_to_process.append(f)
            logger.info(f"Found: {f}")
        else:
            logger.info(f"Not found: {f}")

    if not files_to_process:
        logger.error("No files found to enhance")
        return

    logger.info(f"Processing {len(files_to_process)} files...")

    success_count = 0
    for file_path in files_to_process:
        try:
            # Generate tests
            if generate_tests(file_path):
                success_count += 1

            # Add documentation
            add_documentation(file_path)

            # Refactor code
            refactor_code(file_path)

            # Optimize performance
            optimize_performance(file_path)

            # Add type hints
            add_type_hints(file_path)

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info("=" * 60)
    logger.info("Enhancement Complete!")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Files processed: {len(files_to_process)}")
    logger.info(f"  Enhancements applied: {success_count * 5}")  # 5 enhancements per file
    logger.info("  Cost: $0 ✅")
    logger.info("Check these directories for generated files:")
    logger.info("  - tests/")
    logger.info(
        "  - Gemini Enhancement/ (look for _documented, _refactored, _optimized, _typed versions)"
    )


if __name__ == "__main__":
    main()
