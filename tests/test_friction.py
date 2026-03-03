"""Tests for friction factor models"""

import pytest
from python_magnetcooling.friction import (
    FrictionModel,
    ConstantFriction,
    BlasiusFriction,
    FilonenkoFriction,
    ColebrookFriction,
    SwameeFriction,
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


class TestFilonenkoFriction:
    """Test Filonenko friction correlation"""

    def test_filonenko_turbulent(self):
        """Test Filonenko gives a positive, physically reasonable f for turbulent flow"""
        model = FilonenkoFriction()
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)
        assert f > 0.01
        assert f < 0.1

    def test_filonenko_decreases_with_reynolds(self):
        """Test that Filonenko friction factor decreases with increasing Re"""
        model = FilonenkoFriction()
        f_low = model.compute(reynolds=5000, hydraulic_diameter=0.01)
        f_high = model.compute(reynolds=100000, hydraulic_diameter=0.01)
        assert f_low > f_high

    def test_filonenko_laminar_fallback(self):
        """Test laminar flow fallback: f = 64/Re for Re < 2300"""
        model = FilonenkoFriction()
        re = 1000
        f = model.compute(reynolds=re, hydraulic_diameter=0.01)
        assert abs(f - 64.0 / re) < 1e-9

    def test_filonenko_formula(self):
        """Test Filonenko formula: f = 1/(1.82·log10(Re) - 1.64)²"""
        from math import log10
        model = FilonenkoFriction()
        re = 10000
        f = model.compute(reynolds=re, hydraulic_diameter=0.01)
        expected = 1.0 / (1.82 * log10(re) - 1.64) ** 2
        assert abs(f - expected) < 1e-9


class TestColebrookFriction:
    """Test Colebrook friction correlation"""

    def test_colebrook_turbulent(self):
        """Test Colebrook gives a positive, physically reasonable f for turbulent flow"""
        model = ColebrookFriction()
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)
        assert f > 0.01
        assert f < 0.1

    def test_colebrook_decreases_with_reynolds(self):
        """Test that Colebrook friction decreases with Re for smooth pipes"""
        model = ColebrookFriction(roughness=0.0)  # smooth pipe limit
        f_low = model.compute(reynolds=5000, hydraulic_diameter=0.01)
        f_high = model.compute(reynolds=100000, hydraulic_diameter=0.01)
        assert f_low > f_high

    def test_colebrook_laminar_fallback(self):
        """Test laminar flow fallback: f = 64/Re for Re < 2300"""
        model = ColebrookFriction()
        re = 1000
        f = model.compute(reynolds=re, hydraulic_diameter=0.01)
        assert abs(f - 64.0 / re) < 1e-9

    def test_colebrook_roughness_increases_friction(self):
        """Test that rougher pipe gives higher friction factor"""
        model_smooth = ColebrookFriction(roughness=0.001e-3)
        model_rough = ColebrookFriction(roughness=0.5e-3)
        re = 50000
        dh = 0.01
        f_smooth = model_smooth.compute(reynolds=re, hydraulic_diameter=dh)
        f_rough = model_rough.compute(reynolds=re, hydraulic_diameter=dh)
        assert f_rough > f_smooth


class TestSwameeFriction:
    """Test Swamee-Jain friction correlation"""

    def test_swamee_turbulent(self):
        """Test Swamee gives a positive, physically reasonable f for turbulent flow"""
        model = SwameeFriction()
        f = model.compute(reynolds=10000, hydraulic_diameter=0.01)
        assert f > 0.01
        assert f < 0.1

    def test_swamee_laminar_fallback(self):
        """Test laminar flow fallback: f = 64/Re for Re < 2300"""
        model = SwameeFriction()
        re = 1000
        f = model.compute(reynolds=re, hydraulic_diameter=0.01)
        assert abs(f - 64.0 / re) < 1e-9

    def test_swamee_close_to_colebrook(self):
        """Test that Swamee-Jain is within 1% of Colebrook (by design)"""
        roughness = 0.012e-3
        swamee = SwameeFriction(roughness=roughness)
        colebrook = ColebrookFriction(roughness=roughness)
        dh = 0.01

        for re in [5000, 10000, 50000, 100000]:
            f_swamee = swamee.compute(reynolds=re, hydraulic_diameter=dh)
            f_colebrook = colebrook.compute(reynolds=re, hydraulic_diameter=dh)
            relative_diff = abs(f_swamee - f_colebrook) / f_colebrook
            assert relative_diff < 0.02, (
                f"Swamee and Colebrook differ by {relative_diff:.1%} at Re={re}"
            )

    def test_swamee_roughness_increases_friction(self):
        """Test that rougher pipe gives higher friction factor"""
        model_smooth = SwameeFriction(roughness=0.001e-3)
        model_rough = SwameeFriction(roughness=0.5e-3)
        re = 50000
        dh = 0.01
        f_smooth = model_smooth.compute(reynolds=re, hydraulic_diameter=dh)
        f_rough = model_rough.compute(reynolds=re, hydraulic_diameter=dh)
        assert f_rough > f_smooth
