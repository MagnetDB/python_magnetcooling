"""Test basic package imports"""

import subprocess
import sys

import pytest

# Imported in a fresh interpreter: a module cached by an earlier test cannot
# hide an import error.
FRESH_IMPORT_MODULES = [
    "python_magnetcooling",
    "python_magnetcooling.waterflow",
    "python_magnetcooling.waterflow_factory",
    "python_magnetcooling.fitting",
    "python_magnetcooling.heatexchanger_primary",
    "python_magnetcooling.thermohydraulics",
]


def _run_fresh(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with the same sys.path."""
    return subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path[:0] = {sys.path!r}\n{code}"],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("module", FRESH_IMPORT_MODULES)
def test_fresh_import(module):
    """Each public module imports in a fresh interpreter."""
    result = _run_fresh(f"import {module}")
    assert result.returncode == 0, result.stderr


def test_import_adds_only_pint_scoped_warning_filters():
    """Importing the package must not silence warnings of other libraries.

    Records the ``warnings.filterwarnings``/``simplefilter`` calls made from
    ``python_magnetcooling`` modules: diffing ``warnings.filters`` would also
    catch the dependencies' own filters, and misses a filter that is already
    present (``warnings`` moves it to the front instead of adding it).
    """
    result = _run_fresh(
        "import warnings\n"
        "calls = []\n"
        "def record(real):\n"
        "    def wrapper(action, *args, **kwargs):\n"
        "        caller = sys._getframe(1).f_globals.get('__name__', '')\n"
        "        if caller.startswith('python_magnetcooling'):\n"
        "            calls.append((real.__name__, action, kwargs.get('module', '')))\n"
        "        return real(action, *args, **kwargs)\n"
        "    return wrapper\n"
        "warnings.filterwarnings = record(warnings.filterwarnings)\n"
        "warnings.simplefilter = record(warnings.simplefilter)\n"
        f"for m in {FRESH_IMPORT_MODULES!r}:\n"
        "    __import__(m)\n"
        "bad = [c for c in calls if c[1] == 'ignore' and 'pint' not in c[2]]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    assert result.returncode == 0, f"non-pint ignore filters: {result.stdout}{result.stderr}"


class TestPackageImports:
    """Test that all modules can be imported"""

    def test_import_package(self):
        """Test importing the main package"""
        import python_magnetcooling

        assert python_magnetcooling is not None

    def test_import_exceptions(self):
        """Test importing exceptions module"""
        from python_magnetcooling import exceptions

        assert exceptions is not None

    def test_import_water_properties(self):
        """Test importing water_properties module"""
        from python_magnetcooling import water_properties

        assert water_properties is not None

    def test_import_correlations(self):
        """Test importing correlations module"""
        from python_magnetcooling import correlations

        assert correlations is not None

    def test_import_friction(self):
        """Test importing friction module"""
        from python_magnetcooling import friction

        assert friction is not None

    def test_import_version(self):
        """Test importing version module"""
        from python_magnetcooling import version

        assert version is not None


class TestExceptionClasses:
    """Test that exception classes are accessible"""

    def test_exception_classes_importable(self):
        """Test that all exception classes can be imported"""
        from python_magnetcooling.exceptions import (
            ConvergenceError,
            CorrelationError,
            FrictionError,
            HeatExchangerError,
            InvalidNTUError,
            MagnetCoolingError,
            ValidationError,
            WaterPropertiesError,
        )

        assert MagnetCoolingError is not None
        assert WaterPropertiesError is not None
        assert CorrelationError is not None
        assert FrictionError is not None
        assert ValidationError is not None
        assert ConvergenceError is not None
        assert HeatExchangerError is not None
        assert InvalidNTUError is not None
