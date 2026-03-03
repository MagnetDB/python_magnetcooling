"""Tests for friction factor models"""

import pytest
from python_magnetcooling.friction import (
    FrictionModel,
    ConstantFriction,
    BlasiusFriction,
    KarmanFriction,
    get_friction_model,
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


class TestKarmanFriction:
    """Test Karman-Nikuradse friction correlation"""

    def test_karman_friction_calculation(self):
        """Test Karman correlation converges"""
        model = KarmanFriction()
        reynolds = 50000

        f = model.compute(reynolds=reynolds, hydraulic_diameter=0.01)

        # Should give reasonable turbulent friction factor
        assert 0.01 < f < 0.05

    def test_karman_decreases_with_reynolds(self):
        """Test that Karman friction decreases with Reynolds number"""
        model = KarmanFriction()

        f_low = model.compute(reynolds=10000, hydraulic_diameter=0.01)
        f_high = model.compute(reynolds=100000, hydraulic_diameter=0.01)

        assert f_low > f_high

    def test_karman_laminar_flow(self):
        """Test Karman handles laminar flow correctly"""
        model = KarmanFriction()

        f = model.compute(reynolds=1500, hydraulic_diameter=0.01)

        # Laminar: f = 64/Re
        expected = 64.0 / 1500
        assert abs(f - expected) < 1e-6

    def test_karman_convergence(self):
        """Test that Karman iteration converges"""
        model = KarmanFriction()

        # Should converge without raising exception
        reynolds_values = [10000, 50000, 100000, 500000]
        for re in reynolds_values:
            f = model.compute(reynolds=re, hydraulic_diameter=0.01)
            assert f > 0  # Valid result

    def test_karman_similar_to_blasius(self):
        """Test that Karman gives similar results to Blasius for smooth pipes"""
        karman = KarmanFriction()
        blasius = BlasiusFriction()

        reynolds = 50000
        f_karman = karman.compute(reynolds=reynolds, hydraulic_diameter=0.01)
        f_blasius = blasius.compute(reynolds=reynolds, hydraulic_diameter=0.01)

        # Should be within 10% for smooth pipes in turbulent range
        relative_diff = abs(f_karman - f_blasius) / f_blasius
        assert relative_diff < 0.1


class TestFrictionModelRegistry:
    """Test friction model registry and get_friction_model function"""

    def test_get_karman_model(self):
        """Test getting Karman model from registry"""
        model = get_friction_model('Karman')
        assert isinstance(model, KarmanFriction)

    def test_all_models_available(self):
        """Test that all models are in available models list"""
        from python_magnetcooling.friction import available_friction_models

        models = available_friction_models()
        assert 'Karman' in models
        assert 'Constant' in models
        assert 'Blasius' in models
        assert 'Rough' in models
