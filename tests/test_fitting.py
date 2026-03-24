"""
Test fitting module - hydraulic system curve fitting.

This test suite validates the complete fitting pipeline from raw arrays
to WaterFlow objects, including:
- Individual fitting functions (pump speed, flow rate, pressure)
- Validation and error handling
- Complete orchestration pipeline
- Integration with WaterFlow and waterflow_factory
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from python_magnetcooling.fitting import (
    FitResult,
    PumpSpeedFit,
    FlowPressureFit,
    fit_pump_speed_simple,
    fit_pump_speed_piecewise,
    fit_flow_rate,
    fit_pressure,
    compute_back_pressure_stats,
    fit_hydraulic_system,
    build_waterflow,
)


# =============================================================================
# Test Data Generator
# =============================================================================

# Known parameters used to generate synthetic data
KNOWN_VPMAX = 2840.0  # rpm
KNOWN_VP0 = 1000.0    # rpm
KNOWN_IMAX = 28000.0  # A
KNOWN_F0 = 0.0        # l/s
KNOWN_FMAX = 140.0    # l/s
KNOWN_PMIN = 4.0      # bar
KNOWN_PMAX = 22.0     # bar
KNOWN_BP = 4.0        # bar


def make_synthetic_data(n=200, noise_level=0.01, seed=42, include_plateau=False):
    """
    Generate synthetic hydraulic data with optional noise.
    
    Parameters
    ----------
    n : int
        Number of data points.
    noise_level : float
        Relative noise level (0.02 = 2% noise).
    seed : int
        Random seed for reproducibility.
    include_plateau : bool
        If True, add plateau region beyond Imax for piecewise testing.
    
    Returns
    -------
    dict
        Dictionary with current and measurement arrays.
    """
    rng = np.random.default_rng(seed)
    
    if include_plateau:
        # Include data beyond Imax to test piecewise fitting
        current = np.linspace(300, KNOWN_IMAX * 1.2, n)
    else:
        current = np.linspace(300, KNOWN_IMAX, n)
    
    # True models
    vp = np.where(
        current <= KNOWN_IMAX,
        KNOWN_VPMAX * (current / KNOWN_IMAX) ** 2 + KNOWN_VP0,
        KNOWN_VPMAX + KNOWN_VP0  # Saturated
    )
    
    vp_ratio = vp / (KNOWN_VPMAX + KNOWN_VP0)
    flow = KNOWN_F0 + KNOWN_FMAX * vp_ratio
    pressure = KNOWN_PMIN + KNOWN_PMAX * vp_ratio ** 2
    back_pressure = np.full(n, KNOWN_BP)
    
    # Add noise
    if noise_level > 0:
        vp += rng.normal(0, noise_level * KNOWN_VPMAX, n)
        flow += rng.normal(0, noise_level * KNOWN_FMAX, n)
        pressure += rng.normal(0, noise_level * KNOWN_PMAX, n)
        back_pressure += rng.normal(0, 0.1, n)
    
    return {
        "current": current,
        "pump_speed": vp,
        "flow": flow,
        "pressure": pressure,
        "back_pressure": back_pressure,
    }


# =============================================================================
# Test Pump Speed Fitting - Simple Method
# =============================================================================


def test_fit_pump_speed_simple_noiseless():
    """Test simple fit recovers exact parameters from noiseless data."""
    data = make_synthetic_data(n=100, noise_level=0.0)
    
    fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    assert isinstance(fit, PumpSpeedFit)
    assert fit.imax == KNOWN_IMAX
    assert fit.imax_detected is False
    
    # Should recover exact parameters (tolerance < 1e-6)
    assert abs(fit.vpmax - KNOWN_VPMAX) < 1e-6
    assert abs(fit.vp0 - KNOWN_VP0) < 1e-6
    
    # R² should be nearly perfect
    assert fit.fit_result.r_squared > 0.99999
    
    # Verify pump_speed method works
    speed_at_14kA = fit.pump_speed(14000)
    expected = KNOWN_VPMAX * (14000 / KNOWN_IMAX) ** 2 + KNOWN_VP0
    assert abs(speed_at_14kA - expected) < 1e-6


def test_fit_pump_speed_simple_noisy():
    """Test simple fit recovers parameters within 2% from noisy data."""
    data = make_synthetic_data(n=200, noise_level=0.02)
    
    fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Parameters should be recovered within 2%
    vpmax_error = abs(fit.vpmax - KNOWN_VPMAX) / KNOWN_VPMAX
    vp0_error = abs(fit.vp0 - KNOWN_VP0) / KNOWN_VP0
    
    assert vpmax_error < 0.02, f"Vpmax error {vpmax_error*100:.2f}% exceeds 2%"
    assert vp0_error < 0.02, f"Vp0 error {vp0_error*100:.2f}% exceeds 2%"
    
    # R² should still be high
    assert fit.fit_result.r_squared > 0.98


def test_fit_pump_speed_simple_invalid_imax():
    """Test that invalid imax raises ValueError."""
    data = make_synthetic_data(n=50)
    
    with pytest.raises(ValueError, match="imax must be positive"):
        fit_pump_speed_simple(data["current"], data["pump_speed"], imax=-1000)
    
    with pytest.raises(ValueError, match="imax must be positive"):
        fit_pump_speed_simple(data["current"], data["pump_speed"], imax=0)


# =============================================================================
# Test Pump Speed Fitting - Piecewise Method
# =============================================================================


def test_fit_pump_speed_piecewise_import_error():
    """Test that missing pwlf raises informative ImportError."""
    data = make_synthetic_data(n=50)
    
    # Try to fit - should raise ImportError if pwlf not installed
    try:
        fit = fit_pump_speed_piecewise(data["current"], data["pump_speed"])
        # If pwlf IS installed, check it works
        assert isinstance(fit, PumpSpeedFit)
    except ImportError as e:
        # Verify error message is helpful
        assert "pwlf is required" in str(e)
        assert "pip install pwlf" in str(e)


def test_fit_pump_speed_piecewise_single_segment():
    """Test piecewise fit with one segment (no breakpoint detection)."""
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    try:
        fit = fit_pump_speed_piecewise(
            data["current"],
            data["pump_speed"],
            max_segments=1
        )
        
        assert isinstance(fit, PumpSpeedFit)
        # With 1 segment, Imax should be the data maximum
        assert fit.imax_detected is False
        assert fit.imax == pytest.approx(data["current"].max(), rel=1e-6)
        
    except ImportError:
        pytest.skip("pwlf not installed")


def test_fit_pump_speed_piecewise_two_segments():
    """Test piecewise fit with breakpoint detection."""
    # Generate data with plateau beyond Imax
    data = make_synthetic_data(n=150, noise_level=0.02, include_plateau=True)
    
    try:
        fit = fit_pump_speed_piecewise(
            data["current"],
            data["pump_speed"],
            max_segments=2
        )
        
        assert isinstance(fit, PumpSpeedFit)
        
        # With 2 segments and plateau, should detect Imax
        if fit.imax_detected:
            # Detected Imax should be within 5% of true value
            imax_error = abs(fit.imax - KNOWN_IMAX) / KNOWN_IMAX
            assert imax_error < 0.05, f"Imax detection error {imax_error*100:.1f}% > 5%"
            
            # Should have breakpoints
            assert fit.breakpoints is not None
            assert len(fit.breakpoints) >= 2
        
    except ImportError:
        pytest.skip("pwlf not installed")


# =============================================================================
# Test Flow Rate and Pressure Fitting
# =============================================================================


def test_fit_flow_rate():
    """Test flow rate fitting with known pump parameters."""
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    # First fit pump speed
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Then fit flow rate
    flow_fit = fit_flow_rate(
        data["current"],
        data["flow"],
        pump_fit
    )
    
    assert isinstance(flow_fit, FitResult)
    assert len(flow_fit.parameters) == 2
    
    f0, fmax = flow_fit.parameters
    
    # Should recover F0 and Fmax within 2%
    f0_error = abs(f0 - KNOWN_F0)
    fmax_error = abs(fmax - KNOWN_FMAX) / KNOWN_FMAX
    
    assert f0_error < 1.0, f"F0 error {f0_error:.2f} l/s too large"
    assert fmax_error < 0.02, f"Fmax error {fmax_error*100:.2f}% > 2%"
    
    # R² should be high
    assert flow_fit.r_squared > 0.98


def test_fit_pressure():
    """Test pressure fitting with known pump parameters."""
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    # First fit pump speed
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Then fit pressure
    pressure_fit = fit_pressure(
        data["current"],
        data["pressure"],
        pump_fit
    )
    
    assert isinstance(pressure_fit, FitResult)
    assert len(pressure_fit.parameters) == 2
    
    pmin, pmax = pressure_fit.parameters
    
    # Should recover Pmin and Pmax within 2%
    pmin_error = abs(pmin - KNOWN_PMIN) / KNOWN_PMIN
    pmax_error = abs(pmax - KNOWN_PMAX) / KNOWN_PMAX
    
    assert pmin_error < 0.02, f"Pmin error {pmin_error*100:.2f}% > 2%"
    assert pmax_error < 0.02, f"Pmax error {pmax_error*100:.2f}% > 2%"
    
    # R² should be high
    assert pressure_fit.r_squared > 0.98


# =============================================================================
# Test Back Pressure Statistics
# =============================================================================


def test_compute_back_pressure_stats():
    """Test back pressure statistics calculation."""
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    mean, std = compute_back_pressure_stats(data["back_pressure"])
    
    # Mean should be close to true value
    assert abs(mean - KNOWN_BP) < 0.2
    
    # Std should be reasonable (noise is 0.1 bar)
    assert std > 0
    assert std < 0.3


def test_compute_back_pressure_stats_invalid_input():
    """Test that invalid inputs raise ValueError."""
    # Empty array
    with pytest.raises(ValueError, match="empty"):
        compute_back_pressure_stats(np.array([]))
    
    # Array with NaN
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_back_pressure_stats(np.array([4.0, np.nan, 4.1]))
    
    # Not a numpy array
    with pytest.raises(ValueError, match="numpy array"):
        compute_back_pressure_stats([4.0, 4.1, 4.0])


# =============================================================================
# Test Complete Pipeline (fit_hydraulic_system)
# =============================================================================


def test_fit_hydraulic_system_simple():
    """Test end-to-end fitting with simple method."""
    data = make_synthetic_data(n=200, noise_level=0.02)
    
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple",
        current_threshold=300.0
    )
    
    # Check pump fit
    assert isinstance(pump_fit, PumpSpeedFit)
    assert pump_fit.imax == KNOWN_IMAX
    assert pump_fit.imax_detected is False
    
    # Check flow/pressure fit
    assert isinstance(flow_pressure_fit, FlowPressureFit)
    assert abs(flow_pressure_fit.back_pressure - KNOWN_BP) < 0.2
    
    # All parameters should be recovered within tolerance
    assert abs(pump_fit.vpmax - KNOWN_VPMAX) / KNOWN_VPMAX < 0.05
    assert abs(pump_fit.vp0 - KNOWN_VP0) / KNOWN_VP0 < 0.05
    assert abs(flow_pressure_fit.fmax - KNOWN_FMAX) / KNOWN_FMAX < 0.05
    assert abs(flow_pressure_fit.pmin - KNOWN_PMIN) / KNOWN_PMIN < 0.05
    assert abs(flow_pressure_fit.pmax - KNOWN_PMAX) / KNOWN_PMAX < 0.05


def test_fit_hydraulic_system_piecewise():
    """Test end-to-end fitting with piecewise method."""
    data = make_synthetic_data(n=200, noise_level=0.02)
    
    try:
        pump_fit, flow_pressure_fit = fit_hydraulic_system(
            data["current"],
            data["pump_speed"],
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=None,  # Let piecewise detect it
            method="piecewise",
            current_threshold=300.0
        )
        
        assert isinstance(pump_fit, PumpSpeedFit)
        assert isinstance(flow_pressure_fit, FlowPressureFit)
        
    except ImportError:
        pytest.skip("pwlf not installed")


def test_fit_hydraulic_system_invalid_method():
    """Test that invalid method raises ValueError."""
    data = make_synthetic_data(n=50)
    
    with pytest.raises(ValueError, match="method must be one of"):
        fit_hydraulic_system(
            data["current"],
            data["pump_speed"],
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=KNOWN_IMAX,
            method="invalid_method"
        )


def test_fit_hydraulic_system_missing_imax_simple():
    """Test that simple method with no imax raises ValueError."""
    data = make_synthetic_data(n=50)
    
    with pytest.raises(ValueError, match="imax must be provided"):
        fit_hydraulic_system(
            data["current"],
            data["pump_speed"],
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=None,
            method="simple"
        )


# =============================================================================
# Test Data Validation and Filtering
# =============================================================================


def test_current_threshold_filtering():
    """Test that current_threshold filters data correctly."""
    # Generate data starting from 100 A
    rng = np.random.default_rng(42)
    current = np.linspace(100, 28000, 200)
    pump_speed = 2840 * (current / 28000) ** 2 + 1000 + rng.normal(0, 10, 200)
    vp_ratio = pump_speed / (2840 + 1000)
    flow = 140 * vp_ratio
    pressure = 4 + 22 * vp_ratio ** 2
    back_pressure = np.full(200, 4.0)
    
    # Fit with threshold=500 should filter out low current points
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        current, pump_speed, flow, pressure, back_pressure,
        imax=28000,
        method="simple",
        current_threshold=500.0
    )
    
    # Should succeed with filtered data
    assert isinstance(pump_fit, PumpSpeedFit)
    
    # If threshold is too high, should fail
    with pytest.raises(ValueError, match="No data points remain"):
        fit_hydraulic_system(
            current, pump_speed, flow, pressure, back_pressure,
            imax=28000,
            method="simple",
            current_threshold=30000.0  # Higher than all data
        )


def test_validation_array_length_mismatch():
    """Test that mismatched array lengths raise ValueError."""
    data = make_synthetic_data(n=100)
    
    # Different length arrays
    with pytest.raises(ValueError, match="Array length mismatch"):
        fit_hydraulic_system(
            data["current"],
            data["pump_speed"][:50],  # Different length
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=KNOWN_IMAX,
            method="simple"
        )


def test_validation_nan_values():
    """Test that NaN values in arrays raise ValueError."""
    data = make_synthetic_data(n=100)
    
    # Inject NaN
    bad_pump_speed = data["pump_speed"].copy()
    bad_pump_speed[10] = np.nan
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        fit_hydraulic_system(
            data["current"],
            bad_pump_speed,
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=KNOWN_IMAX,
            method="simple"
        )


def test_validation_insufficient_points():
    """Test that too few data points raise ValueError."""
    data = make_synthetic_data(n=2)  # Only 2 points
    
    with pytest.raises(ValueError, match="Insufficient data points"):
        fit_hydraulic_system(
            data["current"],
            data["pump_speed"],
            data["flow"],
            data["pressure"],
            data["back_pressure"],
            imax=KNOWN_IMAX,
            method="simple"
        )


# =============================================================================
# Test WaterFlow Construction
# =============================================================================


def test_build_waterflow():
    """Test building WaterFlow from fit results."""
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple"
    )
    
    waterflow = build_waterflow(pump_fit, flow_pressure_fit)
    
    # Check WaterFlow attributes are set correctly
    assert waterflow.pump_speed_min == pytest.approx(pump_fit.vp0, rel=1e-6)
    assert waterflow.pump_speed_max == pytest.approx(pump_fit.vpmax, rel=1e-6)
    assert waterflow.flow_min == pytest.approx(flow_pressure_fit.f0, rel=1e-6)
    assert waterflow.flow_max == pytest.approx(flow_pressure_fit.fmax, rel=1e-6)
    assert waterflow.pressure_min == pytest.approx(flow_pressure_fit.pmin, rel=1e-6)
    assert waterflow.pressure_max == pytest.approx(flow_pressure_fit.pmax, rel=1e-6)
    assert waterflow.pressure_back == pytest.approx(flow_pressure_fit.back_pressure, rel=1e-6)
    assert waterflow.current_max == pytest.approx(pump_fit.imax, rel=1e-6)
    
    # Test WaterFlow methods produce reasonable values
    flow_at_20kA = waterflow.flow_rate(20000)
    pressure_at_20kA = waterflow.pressure(20000)
    
    assert flow_at_20kA > 0
    assert pressure_at_20kA > 0


def test_waterflow_factory_from_fits():
    """Test waterflow_factory.from_fits() integration."""
    from python_magnetcooling.waterflow_factory import from_fits
    
    data = make_synthetic_data(n=100, noise_level=0.01)
    
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple"
    )
    
    # Test from_fits factory method
    waterflow = from_fits(pump_fit, flow_pressure_fit)
    
    assert waterflow.pump_speed_max == pytest.approx(pump_fit.vpmax, rel=1e-6)
    assert waterflow.flow_max == pytest.approx(flow_pressure_fit.fmax, rel=1e-6)
    assert waterflow.current_max == pytest.approx(pump_fit.imax, rel=1e-6)


# =============================================================================
# Test Fit Quality Metrics
# =============================================================================


def test_fit_result_r_squared_perfect():
    """Test R² is 1.0 for noiseless data."""
    data = make_synthetic_data(n=100, noise_level=0.0)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # R² should be nearly perfect for noiseless data
    assert pump_fit.fit_result.r_squared > 0.99999


def test_fit_result_r_squared_noisy():
    """Test R² is less than 1.0 for noisy data."""
    data = make_synthetic_data(n=100, noise_level=0.05)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # R² should be less than 1.0 but still high
    assert 0.90 < pump_fit.fit_result.r_squared < 1.0


def test_fit_result_residuals():
    """Test residuals are computed correctly."""
    data = make_synthetic_data(n=50, noise_level=0.01)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Residuals should have same length as input
    assert len(pump_fit.fit_result.residuals) == len(data["current"])
    
    # Mean of residuals should be close to zero
    assert abs(np.mean(pump_fit.fit_result.residuals)) < 10


# =============================================================================
# Test Edge Cases
# =============================================================================


def test_pump_speed_at_imax():
    """Test pump_speed() method at exactly Imax."""
    data = make_synthetic_data(n=50)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # At Imax, should return vpmax + vp0
    speed_at_imax = pump_fit.pump_speed(KNOWN_IMAX)
    expected = pump_fit.vpmax + pump_fit.vp0
    
    assert abs(speed_at_imax - expected) < 1e-6


def test_pump_speed_above_imax():
    """Test pump_speed() method above Imax (should saturate)."""
    data = make_synthetic_data(n=50)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Above Imax, should return vpmax + vp0 (saturated)
    speed_above_imax = pump_fit.pump_speed(KNOWN_IMAX * 1.5)
    expected = pump_fit.vpmax + pump_fit.vp0
    
    assert abs(speed_above_imax - expected) < 1e-6


def test_dataclass_immutability():
    """Test that result dataclasses are immutable (frozen)."""
    data = make_synthetic_data(n=50)
    
    pump_fit = fit_pump_speed_simple(
        data["current"],
        data["pump_speed"],
        imax=KNOWN_IMAX
    )
    
    # Should not be able to modify frozen dataclass
    with pytest.raises(Exception):  # FrozenInstanceError
        pump_fit.vpmax = 9999


# =============================================================================
# Integration Tests
# =============================================================================


def test_complete_workflow_simple():
    """Test complete workflow: data → fit → WaterFlow → calculations."""
    # Generate data
    data = make_synthetic_data(n=200, noise_level=0.02)
    
    # Fit all curves
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple"
    )
    
    # Build WaterFlow
    waterflow = build_waterflow(pump_fit, flow_pressure_fit)
    
    # Perform calculations
    test_currents = [10000, 15000, 20000, 25000]
    
    for current in test_currents:
        flow = waterflow.flow_rate(current)
        pressure = waterflow.pressure(current)
        
        # Values should be reasonable
        assert flow > 0
        assert pressure > 0
        assert pressure >= waterflow.pressure_min
        assert pressure <= waterflow.pressure_max


def test_multiple_fits_reproducible():
    """Test that fitting is reproducible with same data."""
    data = make_synthetic_data(n=100, noise_level=0.02, seed=123)
    
    # Fit twice
    pump_fit1, flow_pressure_fit1 = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple"
    )
    
    pump_fit2, flow_pressure_fit2 = fit_hydraulic_system(
        data["current"],
        data["pump_speed"],
        data["flow"],
        data["pressure"],
        data["back_pressure"],
        imax=KNOWN_IMAX,
        method="simple"
    )
    
    # Results should be identical
    assert pump_fit1.vpmax == pytest.approx(pump_fit2.vpmax)
    assert pump_fit1.vp0 == pytest.approx(pump_fit2.vp0)
    assert flow_pressure_fit1.fmax == pytest.approx(flow_pressure_fit2.fmax)
    assert flow_pressure_fit1.pmax == pytest.approx(flow_pressure_fit2.pmax)
