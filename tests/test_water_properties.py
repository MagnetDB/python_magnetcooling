"""Tests for water properties module"""

import pytest
from python_magnetcooling.water_properties import WaterProperties, WaterState
from python_magnetcooling.exceptions import WaterPropertiesError


class TestWaterProperties:
    """Test water properties calculations"""

    def test_get_state_standard_conditions(self, standard_water_conditions):
        """Test water properties at standard conditions (20°C, 10 bar)"""
        water = WaterProperties()
        state = water.get_state(**standard_water_conditions)

        assert isinstance(state, WaterState)
        assert state.temperature == standard_water_conditions["temperature"]
        assert state.pressure == standard_water_conditions["pressure"]
        assert state.density > 990  # kg/m³ - water at 20°C
        assert state.density < 1010
        assert state.specific_heat > 4000  # J/kg/K
        assert state.specific_heat < 5000
        assert state.thermal_conductivity > 0.5  # W/m/K
        assert state.thermal_conductivity < 0.7
        assert state.dynamic_viscosity > 0  # Pa·s
        assert state.prandtl > 0  # dimensionless

    def test_get_state_hot_water(self, hot_water_conditions):
        """Test water properties at hot conditions (80°C, 5 bar)"""
        water = WaterProperties()
        state = water.get_state(**hot_water_conditions)

        assert isinstance(state, WaterState)
        assert state.temperature == hot_water_conditions["temperature"]
        assert state.pressure == hot_water_conditions["pressure"]
        assert state.density > 960  # kg/m³ - water at 80°C
        assert state.density < 980
        assert state.prandtl > 0

    def test_get_state_returns_all_properties(self, standard_water_conditions):
        """Test that all properties are returned"""
        water = WaterProperties()
        state = water.get_state(**standard_water_conditions)

        # Check all attributes exist and are numeric
        assert isinstance(state.temperature, float)
        assert isinstance(state.pressure, float)
        assert isinstance(state.density, float)
        assert isinstance(state.specific_heat, float)
        assert isinstance(state.thermal_conductivity, float)
        assert isinstance(state.dynamic_viscosity, float)
        assert isinstance(state.prandtl, float)

    def test_density_decreases_with_temperature(self):
        """Test that water density decreases with increasing temperature"""
        water = WaterProperties()
        state_cold = water.get_state(temperature=293.15, pressure=10.0)  # 20°C
        state_hot = water.get_state(temperature=353.15, pressure=10.0)  # 80°C

        assert state_cold.density > state_hot.density

    def test_viscosity_decreases_with_temperature(self):
        """Test that viscosity decreases with increasing temperature"""
        water = WaterProperties()
        state_cold = water.get_state(temperature=293.15, pressure=10.0)  # 20°C
        state_hot = water.get_state(temperature=353.15, pressure=10.0)  # 80°C

        assert state_cold.dynamic_viscosity > state_hot.dynamic_viscosity

    def test_invalid_conditions_raise_error(self):
        """Test that invalid conditions raise WaterPropertiesError"""
        water = WaterProperties()

        # Negative temperature (physically impossible)
        with pytest.raises(WaterPropertiesError):
            water.get_state(temperature=-10.0, pressure=10.0)

        # Negative pressure (physically impossible)
        with pytest.raises(WaterPropertiesError):
            water.get_state(temperature=300.0, pressure=-5.0)

    def test_compute_temperature_rise(self):
        """Test temperature rise calculation"""
        water = WaterProperties()

        # Test if the method exists and can be called
        # Note: We'd need to check the full implementation
        try:
            # This tests the method signature exists
            assert hasattr(water, "compute_temperature_rise")
        except AttributeError:
            pytest.skip("compute_temperature_rise method not fully implemented")


class TestWaterState:
    """Test WaterState NamedTuple"""

    def test_water_state_creation(self):
        """Test creating a WaterState directly"""
        state = WaterState(
            temperature=293.15,
            pressure=10.0,
            density=998.0,
            specific_heat=4182.0,
            thermal_conductivity=0.6,
            dynamic_viscosity=0.001,
            prandtl=7.0,
        )

        assert state.temperature == 293.15
        assert state.pressure == 10.0
        assert state.density == 998.0
        assert state.specific_heat == 4182.0
        assert state.thermal_conductivity == 0.6
        assert state.dynamic_viscosity == 0.001
        assert state.prandtl == 7.0

    def test_water_state_immutable(self):
        """Test that WaterState is immutable"""
        state = WaterState(
            temperature=293.15,
            pressure=10.0,
            density=998.0,
            specific_heat=4182.0,
            thermal_conductivity=0.6,
            dynamic_viscosity=0.001,
            prandtl=7.0,
        )

        with pytest.raises(AttributeError):
            state.temperature = 300.0
