"""Tests for heat transfer correlations"""

import pytest
from python_magnetcooling.correlations import (
    HeatCorrelation,
    MontgomeryCorrelation,
    DittusBoelterCorrelation,
    ColburnCorrelation,
    SilverbergCorrelation,
)
from python_magnetcooling.exceptions import CorrelationError


class TestHeatCorrelation:
    """Test base HeatCorrelation class"""

    def test_compute_nusselt(self):
        """Test generic Nusselt number calculation"""
        # Nu = α·Re^n·Pr^m
        reynolds = 10000.0
        prandtl = 7.0
        alpha = 0.023
        n = 0.8
        m = 0.4

        nusselt = HeatCorrelation.compute_nusselt(reynolds, prandtl, alpha, n, m)

        # Expected: 0.023 * 10000^0.8 * 7^0.4
        # = 0.023 * 2511.886 * 2.292
        expected = 0.023 * (10000.0**0.8) * (7.0**0.4)

        assert nusselt > 0
        assert abs(nusselt - expected) < 1.0  # Close enough

    def test_compute_nusselt_edge_cases(self):
        """Test Nusselt calculation with edge cases"""
        # Low Reynolds number
        nusselt_low = HeatCorrelation.compute_nusselt(1000, 7.0, 0.023, 0.8, 0.4)
        assert nusselt_low > 0

        # High Reynolds number
        nusselt_high = HeatCorrelation.compute_nusselt(100000, 7.0, 0.023, 0.8, 0.4)
        assert nusselt_high > 0
        assert nusselt_high > nusselt_low


class TestMontgomeryCorrelation:
    """Test Montgomery correlation"""

    def test_montgomery_initialization(self):
        """Test Montgomery correlation initialization"""
        corr = MontgomeryCorrelation()
        assert corr.fuzzy_factor == 1.0

        corr_fuzzy = MontgomeryCorrelation(fuzzy_factor=1.2)
        assert corr_fuzzy.fuzzy_factor == 1.2

    def test_montgomery_compute(self, standard_water_conditions, typical_flow_conditions):
        """Test Montgomery heat transfer coefficient calculation"""
        corr = MontgomeryCorrelation()

        h = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        # Heat transfer coefficient should be positive and reasonable
        # Typical values for water: 1000-100000 W/m²/K
        assert h > 0
        assert h > 100  # Minimum reasonable value
        assert h < 1e6  # Maximum reasonable value

    def test_montgomery_increases_with_velocity(self, standard_water_conditions):
        """Test that h increases with velocity"""
        corr = MontgomeryCorrelation()

        h_low = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=1.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )

        h_high = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=5.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )

        assert h_high > h_low

    def test_fuzzy_factor_effect(self, standard_water_conditions, typical_flow_conditions):
        """Test that fuzzy factor affects the result"""
        corr1 = MontgomeryCorrelation(fuzzy_factor=1.0)
        corr2 = MontgomeryCorrelation(fuzzy_factor=1.5)

        h1 = corr1.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )

        h2 = corr2.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )

        # With fuzzy_factor=1.5, result should be 50% higher
        assert abs(h2 / h1 - 1.5) < 0.01


class TestDittusBoelterCorrelation:
    """Test Dittus-Boelter correlation"""

    def test_dittus_compute_turbulent(self, standard_water_conditions, typical_flow_conditions):
        """Test Dittus-Boelter gives a positive, physically reasonable h for turbulent flow"""
        corr = DittusBoelterCorrelation()
        h = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert h > 0
        assert h > 100
        assert h < 1e6

    def test_dittus_raises_for_laminar_flow(self, standard_water_conditions):
        """Test Dittus-Boelter raises CorrelationError for laminar flow (Re < 2300)"""
        corr = DittusBoelterCorrelation()
        with pytest.raises(CorrelationError):
            corr.compute(
                temperature=standard_water_conditions["temperature"],
                pressure=standard_water_conditions["pressure"],
                velocity=0.001,  # very low velocity → laminar
                hydraulic_diameter=0.001,
                length=1.0,
            )

    def test_dittus_fuzzy_factor_scales_result(self, standard_water_conditions, typical_flow_conditions):
        """Test fuzzy_factor scales h proportionally"""
        corr1 = DittusBoelterCorrelation(fuzzy_factor=1.0)
        corr2 = DittusBoelterCorrelation(fuzzy_factor=2.0)

        h1 = corr1.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        h2 = corr2.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert abs(h2 / h1 - 2.0) < 0.01

    def test_dittus_increases_with_velocity(self, standard_water_conditions):
        """Test that h increases with velocity"""
        corr = DittusBoelterCorrelation()
        h_low = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=1.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )
        h_high = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=5.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )
        assert h_high > h_low


class TestColburnCorrelation:
    """Test Colburn correlation"""

    def test_colburn_compute(self, standard_water_conditions, typical_flow_conditions):
        """Test Colburn gives a positive, physically reasonable h"""
        corr = ColburnCorrelation()
        h = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert h > 0
        assert h > 100
        assert h < 1e6

    def test_colburn_fuzzy_factor_scales_result(self, standard_water_conditions, typical_flow_conditions):
        """Test fuzzy_factor scales h proportionally"""
        corr1 = ColburnCorrelation(fuzzy_factor=1.0)
        corr2 = ColburnCorrelation(fuzzy_factor=2.0)

        h1 = corr1.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        h2 = corr2.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert abs(h2 / h1 - 2.0) < 0.01

    def test_colburn_lower_than_dittus(self, standard_water_conditions, typical_flow_conditions):
        """Colburn (Pr^0.3) should give a lower h than Dittus-Boelter (Pr^0.4) for Pr > 1"""
        colburn = ColburnCorrelation()
        dittus = DittusBoelterCorrelation()

        h_colburn = colburn.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        h_dittus = dittus.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        # Water has Pr > 1, so Pr^0.3 < Pr^0.4
        assert h_colburn < h_dittus


class TestSilverbergCorrelation:
    """Test Silverberg correlation"""

    def test_silverberg_compute(self, standard_water_conditions, typical_flow_conditions):
        """Test Silverberg gives a positive, physically reasonable h"""
        corr = SilverbergCorrelation()
        h = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert h > 0
        assert h > 100
        assert h < 1e6

    def test_silverberg_fuzzy_factor_scales_result(self, standard_water_conditions, typical_flow_conditions):
        """Test fuzzy_factor scales h proportionally"""
        corr1 = SilverbergCorrelation(fuzzy_factor=1.0)
        corr2 = SilverbergCorrelation(fuzzy_factor=2.0)

        h1 = corr1.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        h2 = corr2.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            **typical_flow_conditions,
        )
        assert abs(h2 / h1 - 2.0) < 0.01

    def test_silverberg_increases_with_velocity(self, standard_water_conditions):
        """Test that h increases with velocity"""
        corr = SilverbergCorrelation()
        h_low = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=1.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )
        h_high = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=5.0,
            hydraulic_diameter=0.01,
            length=1.0,
        )
        assert h_high > h_low
