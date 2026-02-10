"""Tests for custom exceptions"""

import pytest
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


class TestExceptions:
    """Test custom exception hierarchy"""

    def test_base_exception(self):
        """Test base MagnetCoolingError"""
        with pytest.raises(MagnetCoolingError):
            raise MagnetCoolingError("Test error")

    def test_water_properties_error(self):
        """Test WaterPropertiesError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise WaterPropertiesError("Water properties error")

    def test_correlation_error(self):
        """Test CorrelationError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise CorrelationError("Correlation error")

    def test_friction_error(self):
        """Test FrictionError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise FrictionError("Friction error")

    def test_validation_error(self):
        """Test ValidationError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise ValidationError("Validation error")

    def test_convergence_error(self):
        """Test ConvergenceError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise ConvergenceError("Convergence error")

    def test_heat_exchanger_error(self):
        """Test HeatExchangerError inherits from base"""
        with pytest.raises(MagnetCoolingError):
            raise HeatExchangerError("Heat exchanger error")

    def test_invalid_ntu_error(self):
        """Test InvalidNTUError with parameters"""
        error = InvalidNTUError(
            ntu_value=-1.0,
            tci=293.15,
            thi=313.15,
            pci=10.0,
            phi=5.0,
            debitc=0.5,
            debith=0.3,
        )

        assert error.ntu_value == -1.0
        assert error.tci == 293.15
        assert error.thi == 313.15
        assert error.pci == 10.0
        assert error.phi == 5.0
        assert error.debitc == 0.5
        assert error.debith == 0.3

        with pytest.raises(HeatExchangerError):
            raise error
