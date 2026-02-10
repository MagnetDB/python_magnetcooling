# Testing Guide

## Quick Start

### 1. Install Dependencies

Before running tests, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- Core dependencies: `iapws`, `numpy`, `scipy`, `pandas`, `pint`, `ht`
- Dev dependencies: `pytest`, `pytest-cov`, `black`, `mypy`, `flake8`

### 2. Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_water_properties.py

# Run specific test
pytest tests/test_water_properties.py::TestWaterProperties::test_get_state_standard_conditions

# Run with coverage
pytest --cov=python_magnetcooling --cov-report=html --cov-report=term
```

### 3. View Coverage

After running tests with coverage:

```bash
# Linux
xdg-open htmlcov/index.html

# macOS
open htmlcov/index.html

# Windows
start htmlcov/index.html
```

## Test Structure

```
tests/
├── conftest.py                  # Shared fixtures
├── test_import.py              # Import tests (no dependencies)
├── test_exceptions.py          # Exception hierarchy tests
├── test_water_properties.py    # IAPWS water properties tests
├── test_correlations.py        # Heat transfer correlations
├── test_friction.py            # Friction factor models
└── test_channel.py             # Channel module (placeholder)
```

## Common Issues

### `ModuleNotFoundError: No module named 'iapws'`
**Solution:** Install dependencies with `pip install -e ".[dev]"`

### Import errors in tests
**Solution:** Make sure you're in the project root directory and the package is installed in editable mode

### Tests pass but coverage is low
**Solution:** This is expected initially. Coverage will improve as more tests are added.

## Writing New Tests

1. Create a new file: `tests/test_<module_name>.py`
2. Import required modules and fixtures
3. Create test classes with descriptive names
4. Write test methods starting with `test_`
5. Use fixtures from `conftest.py` for common test data

Example:

```python
import pytest
from python_magnetcooling.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self, standard_water_conditions):
        """Test basic functionality"""
        obj = YourClass()
        result = obj.method(**standard_water_conditions)
        assert result > 0
```

## Test Coverage Goals

- Core modules (water_properties, correlations, friction): > 90%
- Utility modules: > 80%
- Examples and scripts: > 50%

## Continuous Integration

When tests are integrated into CI/CD:
- All tests must pass before merge
- Coverage should not decrease
- New features should include tests
