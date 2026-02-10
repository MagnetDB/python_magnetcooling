"""Test basic package imports"""

import pytest


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
            MagnetCoolingError,
            WaterPropertiesError,
            CorrelationError,
            FrictionError,
            ValidationError,
            ConvergenceError,
            HeatExchangerError,
            InvalidNTUError,
        )

        assert MagnetCoolingError is not None
        assert WaterPropertiesError is not None
        assert CorrelationError is not None
        assert FrictionError is not None
        assert ValidationError is not None
        assert ConvergenceError is not None
        assert HeatExchangerError is not None
        assert InvalidNTUError is not None
