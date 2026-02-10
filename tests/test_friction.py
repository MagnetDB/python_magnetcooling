"""Tests for friction factor models"""

import pytest
from python_magnetcooling.friction import (
    FrictionModel,
    ConstantFriction,
    BlasiusFriction,
)


class TestConstantFriction:
    """Test constant friction model"""

    def test_constant_friction_default(self):
        """Test constant friction with default value"""
        model = ConstantFriction()
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)

        assert f == 0.055  # Default value

    def test_constant_friction_custom_value(self):
        """Test constant friction with custom value"""
        model = ConstantFriction(value=0.02)
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)

        assert f == 0.02

    def test_constant_friction_independent_of_reynolds(self):
        """Test that constant friction doesn't depend on Reynolds number"""
        model = ConstantFriction(value=0.03)

        f1 = model.compute(reynolds=1000, hydraulic_diameter=0.01)
        f2 = model.compute(reynolds=100000, hydraulic_diameter=0.01)

        assert f1 == f2 == 0.03

    def test_roughness_parameter(self):
        """Test that roughness parameter can be set"""
        model = ConstantFriction(value=0.02, roughness=0.05e-3)
        assert model.roughness == 0.05e-3


class TestBlasiusFriction:
    """Test Blasius friction correlation"""

    def test_blasius_friction_calculation(self):
        """Test Blasius correlation: f = 0.316 / Re^0.25"""
        model = BlasiusFriction()
        reynolds = 10000

        f = model.compute(reynolds=reynolds, hydraulic_diameter=0.01)

        # Expected: 0.316 / 10000^0.25 = 0.316 / 10 = 0.0316
        expected = 0.316 / (reynolds**0.25)

        assert abs(f - expected) < 0.0001

    def test_blasius_decreases_with_reynolds(self):
        """Test that Blasius friction decreases with Reynolds number"""
        model = BlasiusFriction()

        f_low = model.compute(reynolds=5000, hydraulic_diameter=0.01)
        f_high = model.compute(reynolds=50000, hydraulic_diameter=0.01)

        assert f_low > f_high

    def test_blasius_typical_values(self):
        """Test Blasius gives reasonable values"""
        model = BlasiusFriction()

        # For turbulent flow in smooth pipes, f typically 0.01-0.05
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)

        assert f > 0.01
        assert f < 0.1

    def test_blasius_at_different_reynolds(self):
        """Test Blasius at various Reynolds numbers"""
        model = BlasiusFriction()

        test_cases = [
            (5000, 0.0376),  # 0.316 / 5000^0.25
            (10000, 0.0316),  # 0.316 / 10000^0.25
            (50000, 0.0211),  # 0.316 / 50000^0.25
            (100000, 0.0178),  # 0.316 / 100000^0.25
        ]

        for reynolds, expected in test_cases:
            f = model.compute(reynolds=reynolds, hydraulic_diameter=0.01)
            assert abs(f - expected) < 0.001


class TestFrictionModel:
    """Test base FrictionModel class"""

    def test_default_roughness(self):
        """Test that default roughness is set correctly"""
        model = ConstantFriction()  # Use concrete class

        # Default roughness for drawn copper: 0.012 mm
        assert model.roughness == 0.012e-3

    def test_custom_roughness(self):
        """Test setting custom roughness"""
        model = ConstantFriction(roughness=0.05e-3)
        assert model.roughness == 0.05e-3
