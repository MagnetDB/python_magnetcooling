"""Tests for heat transfer correlations"""

import pytest
from python_magnetcooling.correlations import (
    HeatCorrelation,
    MontgomeryCorrelation,
    GnielinskiCorrelation,
    get_correlation,
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
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        h2 = corr2.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        # h2 should be 1.5 times h1
        assert h2 > h1
        assert abs(h2 / h1 - 1.5) < 0.01


class TestGnielinskiCorrelation:
    """Test Gnielinski correlation for heat transfer"""

    def test_gnielinski_initialization(self):
        """Test Gnielinski correlation initialization"""
        corr = GnielinskiCorrelation()
        assert corr.fuzzy_factor == 1.0

        corr_fuzzy = GnielinskiCorrelation(fuzzy_factor=1.2)
        assert corr_fuzzy.fuzzy_factor == 1.2

    def test_gnielinski_compute(self, standard_water_conditions, typical_flow_conditions):
        """Test Gnielinski heat transfer coefficient calculation"""
        corr = GnielinskiCorrelation()

        h = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        # Heat transfer coefficient should be positive and reasonable
        assert h > 0
        assert h > 100  # Minimum reasonable value
        assert h < 1e6  # Maximum reasonable value

    def test_gnielinski_reynolds_validation(self, standard_water_conditions):
        """Test that Gnielinski rejects low Reynolds numbers"""
        corr = GnielinskiCorrelation()

        # Low velocity should give low Re
        with pytest.raises(CorrelationError, match="not valid for Re"):
            corr.compute(
                temperature=standard_water_conditions["temperature"],
                pressure=standard_water_conditions["pressure"],
                velocity=0.01,  # Very low velocity
                hydraulic_diameter=0.01,
                length=1.0,
            )

    def test_gnielinski_increases_with_velocity(self, standard_water_conditions):
        """Test that h increases with velocity"""
        corr = GnielinskiCorrelation()

        h_low = corr.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=2.0,
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

    def test_gnielinski_vs_dittus_boelter(self, standard_water_conditions, typical_flow_conditions):
        """Test that Gnielinski gives similar but different results from Dittus-Boelter"""
        from python_magnetcooling.correlations import DittusBoelterCorrelation
        
        gnielinski = GnielinskiCorrelation()
        dittus = DittusBoelterCorrelation()

        h_gn = gnielinski.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        h_db = dittus.compute(
            temperature=standard_water_conditions["temperature"],
            pressure=standard_water_conditions["pressure"],
            velocity=typical_flow_conditions["velocity"],
            hydraulic_diameter=typical_flow_conditions["hydraulic_diameter"],
            length=typical_flow_conditions["length"],
        )

        # Should be in the same ballpark but not identical
        assert h_gn > 0
        assert h_db > 0
        relative_diff = abs(h_gn - h_db) / h_db
        assert 0.01 < relative_diff < 0.5  # Within 1-50% difference


class TestCorrelationRegistry:
    """Test correlation registry and get_correlation function"""

    def test_get_gnielinski_correlation(self):
        """Test getting Gnielinski correlation from registry"""
        corr = get_correlation('Gnielinski')
        assert isinstance(corr, GnielinskiCorrelation)

    def test_all_correlations_available(self):
        """Test that all correlations are in available list"""
        from python_magnetcooling.correlations import available_correlations

        correlations = available_correlations()
        assert 'Gnielinski' in correlations
        assert 'Montgomery' in correlations
        assert 'Dittus' in correlations
        assert 'Colburn' in correlations
        assert 'Silverberg' in correlations

    def test_fuzzy_factor_propagation(self):
        """Test that fuzzy factor is properly set"""
        corr = get_correlation('Gnielinski', fuzzy_factor=1.3)
        assert corr.fuzzy_factor == 1.3

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
