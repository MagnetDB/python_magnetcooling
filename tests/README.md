# Test Suite for python_magnetcooling

This directory contains the test suite for the python_magnetcooling package.

📖 **For detailed testing instructions, see [../TESTING.md](../TESTING.md)**

## Setup

Before running tests, ensure you have installed the package with development dependencies:

```bash
pip install -e ".[dev]"
```

This will install all required dependencies including `iapws`, `numpy`, `scipy`, `pytest`, and `pytest-cov`.

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_water_properties.py
```

### Run with coverage report
```bash
pytest --cov=python_magnetcooling --cov-report=html
```

### Run with verbose output
```bash
pytest -v
```

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_import.py` - Basic package import tests
- `test_exceptions.py` - Custom exception hierarchy tests
- `test_water_properties.py` - Water properties calculation tests
- `test_correlations.py` - Heat transfer correlation tests
- `test_friction.py` - Friction factor model tests
- `test_channel.py` - Channel module tests (placeholder)

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `standard_water_conditions` - 20°C, 10 bar
- `hot_water_conditions` - 80°C, 5 bar
- `typical_flow_conditions` - Standard flow parameters
- `reynolds_turbulent` - Typical turbulent Reynolds number
- `prandtl_water` - Typical Prandtl number for water

## Coverage

Test coverage is configured in `pyproject.toml`. After running tests with coverage,
view the HTML report:

```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Adding New Tests

1. Create a new test file: `test_<module_name>.py`
2. Import the module to test
3. Create test classes with descriptive names
4. Write test methods starting with `test_`
5. Use fixtures from conftest.py when applicable
