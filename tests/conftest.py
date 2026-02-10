"""pytest configuration and fixtures for python_magnetcooling tests"""

import pytest


@pytest.fixture
def standard_water_conditions():
    """Standard test conditions for water: 20°C, 10 bar"""
    return {
        "temperature": 293.15,  # 20°C in Kelvin
        "pressure": 10.0,  # bar
    }


@pytest.fixture
def hot_water_conditions():
    """Hot water test conditions: 80°C, 5 bar"""
    return {
        "temperature": 353.15,  # 80°C in Kelvin
        "pressure": 5.0,  # bar
    }


@pytest.fixture
def typical_flow_conditions():
    """Typical flow conditions for testing"""
    return {
        "velocity": 2.0,  # m/s
        "hydraulic_diameter": 0.01,  # 10 mm
        "length": 1.0,  # m
    }


@pytest.fixture
def reynolds_turbulent():
    """Typical turbulent Reynolds number"""
    return 10000.0


@pytest.fixture
def prandtl_water():
    """Typical Prandtl number for water"""
    return 7.0
