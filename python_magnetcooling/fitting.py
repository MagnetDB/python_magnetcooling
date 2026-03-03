"""
Hydraulic system curve fitting for magnet cooling.

This module provides pure-numerical fitting functions for pump speed, flow rate,
and pressure curves. These functions accept NumPy arrays (not MagnetRun objects,
not TDMS files) and produce typed result dataclasses that feed into WaterFlow
construction.

Physical Models
---------------

1. Pump Speed vs Current:
   Vp(I) = Vpmax · (I/Imax)² + Vp0
   
   Two methods available:
   - Simple quadratic fit (scipy): requires known Imax
   - Piecewise linear fit (pwlf): auto-detects Imax from breakpoint

2. Flow Rate vs Current:
   F(I) = F0 + Fmax · Vp(I)/(Vpmax + Vp0)

3. Pressure vs Current:
   P(I) = Pmin + Pmax · [Vp(I)/(Vpmax + Vp0)]²

4. Back Pressure:
   Not fitted - computed as statistics (mean, std) from measurements

Usage
-----
>>> import numpy as np
>>> from python_magnetcooling.fitting import fit_hydraulic_system, build_waterflow
>>> 
>>> # Your experimental data (NumPy arrays)
>>> current = np.array([...])
>>> pump_speed = np.array([...])
>>> flow_rate = np.array([...])
>>> pressure = np.array([...])
>>> back_pressure = np.array([...])
>>> 
>>> # Fit all curves
>>> pump_fit, flow_pressure_fit = fit_hydraulic_system(
...     current, pump_speed, flow_rate, pressure, back_pressure,
...     imax=28000, method="simple"
... )
>>> 
>>> # Create WaterFlow object
>>> waterflow = build_waterflow(pump_fit, flow_pressure_fit)
>>> 
>>> # Use it
>>> flow_at_20kA = waterflow.flow_rate(20000)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import logging

import numpy as np
from scipy import optimize

logger = logging.getLogger("magnetcooling.fitting")


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass(frozen=True)
class FitResult:
    """
    Base result for any curve fit.

    This dataclass contains all statistical information about a curve fit,
    including fitted parameters, uncertainties, and fit quality metrics.

    Attributes
    ----------
    parameters : np.ndarray
        Fitted parameter values.
    standard_errors : np.ndarray
        Standard errors on each parameter.
    r_squared : float
        Coefficient of determination (R²). Values close to 1.0 indicate
        excellent fit quality.
    residuals : np.ndarray
        Residual array (y_data - y_fit).

    Examples
    --------
    >>> fit = FitResult(
    ...     parameters=np.array([1000.0, 2840.0]),
    ...     standard_errors=np.array([5.0, 10.0]),
    ...     r_squared=0.998,
    ...     residuals=np.array([0.1, -0.2, 0.05])
    ... )
    >>> print(f"R² = {fit.r_squared:.4f}")
    R² = 0.9980
    """

    parameters: np.ndarray
    standard_errors: np.ndarray
    r_squared: float
    residuals: np.ndarray


@dataclass(frozen=True)
class PumpSpeedFit:
    """
    Result of pump speed curve fitting.

    Contains the fitted pump characteristic curve parameters and methods
    to evaluate the pump speed at any current value.

    Attributes
    ----------
    vpmax : float
        Maximum pump speed coefficient [rpm].
    vp0 : float
        Minimum (offset) pump speed [rpm].
    imax : float
        Maximum operating current [A].
        May be input (simple fit) or detected (piecewise fit).
    imax_detected : bool
        True if imax was auto-detected from a piecewise breakpoint.
        Default is False.
    fit_result : FitResult, optional
        Underlying fit statistics (for simple method).
    breakpoints : list of float, optional
        Piecewise breakpoints (only for piecewise method).
    equations : list, optional
        Symbolic piecewise equations (only for piecewise method).

    Examples
    --------
    >>> pump_fit = PumpSpeedFit(
    ...     vpmax=2840.0,
    ...     vp0=1000.0,
    ...     imax=28000.0,
    ...     imax_detected=False
    ... )
    >>> speed_at_20kA = pump_fit.pump_speed(20000)
    >>> print(f"Pump speed at 20 kA: {speed_at_20kA:.1f} rpm")
    """

    vpmax: float
    vp0: float
    imax: float
    imax_detected: bool = False
    fit_result: Optional[FitResult] = None
    breakpoints: Optional[List[float]] = None
    equations: Optional[list] = None

    def pump_speed(self, current: float) -> float:
        """
        Evaluate pump speed at given current.

        Model: Vp(I) = Vpmax·(I/Imax)² + Vp0

        Parameters
        ----------
        current : float
            Operating current [A].

        Returns
        -------
        float
            Pump speed [rpm].

        Notes
        -----
        For currents at or above Imax, returns the maximum pump speed
        (Vpmax + Vp0).

        Examples
        --------
        >>> pump_fit = PumpSpeedFit(vpmax=2840.0, vp0=1000.0, imax=28000.0)
        >>> pump_fit.pump_speed(14000.0)  # Half max current
        1710.0
        >>> pump_fit.pump_speed(28000.0)  # Max current
        3840.0
        """
        if current >= self.imax:
            return self.vpmax + self.vp0
        return self.vpmax * (current / self.imax) ** 2 + self.vp0


@dataclass(frozen=True)
class FlowPressureFit:
    """
    Result of flow rate and pressure curve fitting.

    Contains fitted parameters for both flow rate and pressure curves,
    plus back pressure statistics.

    Attributes
    ----------
    f0 : float
        Flow rate offset [l/s].
    fmax : float
        Flow rate coefficient [l/s].
    pmin : float
        Minimum pressure [bar].
    pmax : float
        Maximum pressure coefficient [bar].
    back_pressure : float
        Mean back pressure [bar].
    back_pressure_std : float
        Standard deviation of back pressure [bar]. Default is 0.0.
    flow_fit : FitResult, optional
        Fit statistics for the flow curve.
    pressure_fit : FitResult, optional
        Fit statistics for the pressure curve.

    Examples
    --------
    >>> flow_pressure_fit = FlowPressureFit(
    ...     f0=0.0,
    ...     fmax=140.0,
    ...     pmin=4.0,
    ...     pmax=22.0,
    ...     back_pressure=4.0,
    ...     back_pressure_std=0.1
    ... )
    >>> print(f"Flow range: {flow_pressure_fit.f0} - {flow_pressure_fit.fmax} l/s")
    """

    f0: float
    fmax: float
    pmin: float
    pmax: float
    back_pressure: float
    back_pressure_std: float = 0.0
    flow_fit: Optional[FitResult] = None
    pressure_fit: Optional[FitResult] = None


@dataclass(frozen=True)
class HysteresisFit:
    """
    Result of hysteresis parameter estimation.

    Contains hysteresis model parameters extracted from power/flow time-series data.
    Each threshold tuple represents (ascending_threshold, descending_threshold) pairs
    that define when the flow rate changes as power increases vs decreases.

    Attributes
    ----------
    thresholds : List[Tuple[float, float]]
        List of (ascending, descending) threshold pairs [MW].
        Ordered from lowest to highest level.
    low_values : List[float]
        Flow rate values for low state at each threshold level [m³/h].
        Length must match thresholds.
    high_values : List[float]
        Flow rate values for high state at each threshold level [m³/h].
        Length must match thresholds.
    diagnostics : dict
        Diagnostic information from fitting process including:
        - n_transitions: Number of transitions detected
        - n_levels_found: Number of distinct levels found
        - n_valid_levels: Number of valid threshold pairs
        - ascending_obs: Observation counts per level (ascending)
        - descending_obs: Observation counts per level (descending)

    Examples
    --------
    >>> hyst_fit = HysteresisFit(
    ...     thresholds=[(3.0, 2.0), (8.0, 6.0), (12.0, 10.0)],
    ...     low_values=[100, 200, 300, 400],
    ...     high_values=[100, 250, 350, 450],
    ...     diagnostics={'n_transitions': 42, 'n_valid_levels': 3}
    ... )
    >>> print(f"Number of levels: {len(hyst_fit.thresholds)}")
    Number of levels: 3

    Notes
    -----
    Use this with build_waterflow_with_hysteresis() to create a WaterFlow
    object with hysteresis support, or access the parameters directly to
    apply the hysteresis model using python_magnetcooling.hysteresis.multi_level_hysteresis().
    """

    thresholds: List[Tuple[float, float]]
    low_values: List[float]
    high_values: List[float]
    diagnostics: dict


@dataclass(frozen=True)
class FitMetrics:
    """
    Fit quality metrics for model evaluation.

    Provides multiple statistical measures to assess how well a fitted model
    matches the raw data. Use these metrics to decide if a refit is necessary.

    Attributes
    ----------
    rmse : float
        Root Mean Square Error - overall fit quality metric.
        Lower is better. Units match the data being fitted.
    mae : float
        Mean Absolute Error - average absolute deviation.
        More robust to outliers than RMSE. Lower is better.
    max_error : float
        Maximum absolute error - worst-case deviation.
        Useful for identifying problematic regions.
    r_squared : float
        Coefficient of determination (R²) - fraction of variance explained.
        Range: -∞ to 1.0. Values close to 1.0 indicate excellent fit.
    mape : float
        Mean Absolute Percentage Error [%].
        Useful for comparing fits across different scales.
        Only computed if all true values are non-zero.
    n_points : int
        Number of data points used in calculation.

    Examples
    --------
    >>> metrics = FitMetrics(rmse=12.5, mae=8.3, max_error=35.2,
    ...                      r_squared=0.985, mape=2.1, n_points=100)
    >>> if metrics.rmse > 20:
    ...     print("Refit recommended: RMSE too high")
    >>> print(f"Model explains {metrics.r_squared*100:.1f}% of variance")
    Model explains 98.5% of variance

    Notes
    -----
    **Interpretation Guidelines:**
    
    - RMSE/MAE: Compare to typical data magnitude. <5% is excellent, <10% is good.
    - R²: >0.95 is excellent, >0.90 is good, <0.80 suggests poor fit.
    - MAPE: <5% is excellent, <10% is good, >20% suggests refit needed.
    - max_error: Check if worst deviation is acceptable for application.
    
    **When to Refit:**
    
    - R² < 0.90 or RMSE > 10% of data range
    - Systematic patterns in residuals (not random scatter)
    - Physical constraints violated (e.g., negative flow rates)
    - New data available that wasn't used in original fit
    """

    rmse: float
    mae: float
    max_error: float
    r_squared: float
    mape: Optional[float]
    n_points: int

    def __str__(self) -> str:
        """Human-readable summary of fit quality."""
        lines = [
            "Fit Quality Metrics:",
            f"  RMSE:      {self.rmse:.4f}",
            f"  MAE:       {self.mae:.4f}",
            f"  Max Error: {self.max_error:.4f}",
            f"  R²:        {self.r_squared:.6f}",
        ]
        if self.mape is not None:
            lines.append(f"  MAPE:      {self.mape:.2f}%")
        lines.append(f"  N points:  {self.n_points}")
        return "\n".join(lines)

    def is_good_fit(
        self, rmse_threshold: float = None, r_squared_threshold: float = 0.90
    ) -> bool:
        """
        Quick check if fit quality is acceptable.

        Parameters
        ----------
        rmse_threshold : float, optional
            Maximum acceptable RMSE. If None, only checks R².
        r_squared_threshold : float, default=0.90
            Minimum acceptable R².

        Returns
        -------
        bool
            True if fit meets quality thresholds.
        """
        if self.r_squared < r_squared_threshold:
            return False
        if rmse_threshold is not None and self.rmse > rmse_threshold:
            return False
        return True


# =============================================================================
# Validation Functions
# =============================================================================


def _validate_array_inputs(
    current: np.ndarray,
    dependent_arrays: dict[str, np.ndarray],
    min_points: int = 3,
) -> None:
    """
    Validate input arrays for fitting.

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    dependent_arrays : dict of str to np.ndarray
        Dictionary mapping variable names to their arrays (e.g.,
        {"pump_speed": pump_speed_array, "flow_rate": flow_rate_array}).
    min_points : int, optional
        Minimum number of data points required. Default is 3.

    Raises
    ------
    ValueError
        If arrays have mismatched lengths, contain NaN/Inf values,
        have insufficient points, or current values are not positive.

    Examples
    --------
    >>> current = np.array([1000, 2000, 3000])
    >>> pump_speed = np.array([1100, 1200, 1300])
    >>> _validate_array_inputs(current, {"pump_speed": pump_speed})
    """
    # Check current array
    if not isinstance(current, np.ndarray):
        raise ValueError(f"current must be a numpy array, got {type(current)}")

    if len(current) < min_points:
        raise ValueError(
            f"Insufficient data points: {len(current)} < {min_points}. "
            f"At least {min_points} points are required for fitting."
        )

    if np.any(~np.isfinite(current)):
        raise ValueError("current array contains NaN or Inf values")

    if np.any(current < 0):
        raise ValueError("current must contain only non-negative values")

    # Check dependent arrays
    for name, array in dependent_arrays.items():
        if not isinstance(array, np.ndarray):
            raise ValueError(f"{name} must be a numpy array, got {type(array)}")

        if len(array) != len(current):
            raise ValueError(
                f"Array length mismatch: current has {len(current)} points, "
                f"{name} has {len(array)} points"
            )

        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} array contains NaN or Inf values")


def _validate_imax(imax: Optional[float], method: str) -> None:
    """
    Validate imax parameter based on fitting method.

    Parameters
    ----------
    imax : float or None
        Maximum operating current [A].
    method : str
        Fitting method ("simple" or "piecewise").

    Raises
    ------
    ValueError
        If imax is required but not provided, or if imax is not positive.

    Examples
    --------
    >>> _validate_imax(28000.0, "simple")  # OK
    >>> _validate_imax(None, "simple")  # Raises ValueError
    >>> _validate_imax(None, "piecewise")  # OK (will be auto-detected)
    """
    if method == "simple" and imax is None:
        raise ValueError(
            "imax must be provided when using method='simple'. "
            "Use method='piecewise' for automatic Imax detection."
        )

    if imax is not None and imax <= 0:
        raise ValueError(f"imax must be positive, got {imax}")


def _validate_method(method: str) -> None:
    """
    Validate fitting method selection.

    Parameters
    ----------
    method : str
        Fitting method to validate.

    Raises
    ------
    ValueError
        If method is not "simple" or "piecewise".

    Examples
    --------
    >>> _validate_method("simple")  # OK
    >>> _validate_method("piecewise")  # OK
    >>> _validate_method("invalid")  # Raises ValueError
    """
    valid_methods = {"simple", "piecewise"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )


def _filter_by_threshold(
    current: np.ndarray,
    threshold: float,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Filter arrays to keep only points where current >= threshold.

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    threshold : float
        Minimum current threshold [A].
    *arrays : np.ndarray
        Additional arrays to filter using the same mask.

    Returns
    -------
    tuple of np.ndarray
        Filtered (current, *arrays).

    Examples
    --------
    >>> current = np.array([100, 500, 1000, 2000])
    >>> pump_speed = np.array([1010, 1050, 1100, 1200])
    >>> filtered_current, filtered_speed = _filter_by_threshold(
    ...     current, 300.0, pump_speed
    ... )
    >>> len(filtered_current)
    3
    """
    mask = current >= threshold
    n_filtered = np.sum(mask)
    n_total = len(current)

    if n_filtered == 0:
        raise ValueError(
            f"No data points remain after filtering with threshold={threshold}. "
            f"All {n_total} current values are below threshold."
        )

    logger.debug(
        f"Filtered {n_total - n_filtered} points below {threshold} A. "
        f"{n_filtered} points remain."
    )

    filtered = [current[mask]]
    filtered.extend(arr[mask] for arr in arrays)
    return tuple(filtered)


# =============================================================================
# Fitting Helper Functions
# =============================================================================


def _compute_fit_statistics(
    y_data: np.ndarray,
    y_fit: np.ndarray,
    params_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute fit quality statistics.

    Parameters
    ----------
    y_data : np.ndarray
        Observed data values.
    y_fit : np.ndarray
        Fitted values from the model.
    params_covariance : np.ndarray
        Covariance matrix from curve_fit.

    Returns
    -------
    parameters : np.ndarray
        Fitted parameter values (from covariance matrix dimensions).
    standard_errors : np.ndarray
        Standard errors on each parameter.
    r_squared : float
        Coefficient of determination (R²).
    residuals : np.ndarray
        Residuals (y_data - y_fit).

    Examples
    --------
    >>> y_data = np.array([1.0, 2.0, 3.0])
    >>> y_fit = np.array([1.1, 1.9, 3.1])
    >>> covariance = np.array([[0.1, 0], [0, 0.2]])
    >>> _, stderr, r2, resid = _compute_fit_statistics(y_data, y_fit, covariance)
    >>> r2 > 0.99
    True
    """
    # Calculate standard errors from covariance matrix
    standard_errors = np.sqrt(np.diag(params_covariance))

    # Calculate residuals
    residuals = y_data - y_fit

    # Calculate R² (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    
    # Avoid division by zero
    if ss_tot == 0:
        r_squared = 1.0 if ss_res == 0 else 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)

    # Extract parameter values (not returned separately, but used in FitResult)
    n_params = len(standard_errors)
    parameters = np.zeros(n_params)  # Placeholder, will be filled by caller

    return parameters, standard_errors, r_squared, residuals


def _extract_piecewise_equations(pwlf_model) -> list:
    """
    Extract symbolic equations from a pwlf piecewise linear fit.

    This is a port of the find_eqn() function from python_magnetrun.
    Requires sympy for symbolic manipulation.

    Parameters
    ----------
    pwlf_model : pwlf.PiecewiseLinFit
        Fitted piecewise linear model.

    Returns
    -------
    list
        List of sympy expressions, one for each segment.

    Raises
    ------
    ImportError
        If sympy is not available.

    Notes
    -----
    This function uses sympy to create symbolic representations of each
    piecewise segment. For a degree-2 polynomial, the equation will be
    of the form: a*x² + b*x + c
    """
    try:
        from sympy import Symbol, symbols
    except ImportError as e:
        raise ImportError(
            "sympy is required for symbolic equation extraction. "
            "Install it with: pip install sympy"
        ) from e

    x = Symbol("x")
    n_segments = pwlf_model.n_segments
    degree = pwlf_model.degree
    
    equations = []
    
    # Extract coefficients for each segment
    # pwlf stores beta coefficients for each polynomial term
    for seg in range(n_segments):
        # Build polynomial equation
        # For degree 2: beta[0] + beta[1]*x + beta[2]*x^2
        equation = 0
        for deg in range(degree + 1):
            idx = seg * (degree + 1) + deg
            if idx < len(pwlf_model.beta):
                coeff = pwlf_model.beta[idx]
                equation += coeff * (x ** deg)
        
        equations.append(equation)
    
    return equations


# =============================================================================
# Pump Speed Fitting Functions
# =============================================================================


def fit_pump_speed_simple(
    current: np.ndarray,
    pump_speed: np.ndarray,
    imax: float,
) -> PumpSpeedFit:
    """
    Fit pump speed vs current using simple quadratic model.

    Model: Vp(I) = Vpmax·(I/Imax)² + Vp0

    Uses scipy.optimize.curve_fit with a fixed Imax value. This is the
    simpler method that requires knowing Imax in advance.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]. Should be filtered (e.g., I >= 300 A).
    pump_speed : np.ndarray
        Pump speed values [rpm].
    imax : float
        Maximum operating current [A]. Must be known/specified.

    Returns
    -------
    PumpSpeedFit
        Fitted parameters and statistics.

    Raises
    ------
    ValueError
        If arrays are invalid or imax is not positive.
    RuntimeError
        If the curve fitting fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> current = np.linspace(1000, 28000, 100)
    >>> pump_speed = 2840 * (current / 28000)**2 + 1000 + np.random.normal(0, 10, 100)
    >>> fit = fit_pump_speed_simple(current, pump_speed, imax=28000)
    >>> print(f"Vpmax = {fit.vpmax:.1f} rpm")
    >>> print(f"Vp0 = {fit.vp0:.1f} rpm")
    >>> print(f"R² = {fit.fit_result.r_squared:.4f}")

    See Also
    --------
    fit_pump_speed_piecewise : Automatic Imax detection using piecewise fitting.
    """
    # Validate inputs
    _validate_array_inputs(current, {"pump_speed": pump_speed})
    if imax <= 0:
        raise ValueError(f"imax must be positive, got {imax}")

    logger.debug(f"Fitting pump speed (simple method) with Imax={imax} A")

    # Define the pump speed model
    def vpump_func(x: np.ndarray, vpmax: float, vp0: float) -> np.ndarray:
        """Vp(I) = Vpmax·(I/Imax)² + Vp0"""
        return vpmax * (x / imax) ** 2 + vp0

    try:
        # Perform curve fitting
        params, params_covariance = optimize.curve_fit(
            vpump_func, current, pump_speed
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Curve fitting failed to converge: {e}. "
            f"Check that your data follows the expected quadratic model."
        ) from e

    vpmax, vp0 = params

    # Compute fit quality
    y_fit = vpump_func(current, vpmax, vp0)
    _, standard_errors, r_squared, residuals = _compute_fit_statistics(
        pump_speed, y_fit, params_covariance
    )

    # Create FitResult
    fit_result = FitResult(
        parameters=params,
        standard_errors=standard_errors,
        r_squared=r_squared,
        residuals=residuals,
    )

    logger.info(
        f"Pump speed fit complete: Vpmax={vpmax:.2f} rpm, Vp0={vp0:.2f} rpm, "
        f"Imax={imax:.0f} A, R²={r_squared:.6f}"
    )

    return PumpSpeedFit(
        vpmax=vpmax,
        vp0=vp0,
        imax=imax,
        imax_detected=False,
        fit_result=fit_result,
    )


def fit_pump_speed_piecewise(
    current: np.ndarray,
    pump_speed: np.ndarray,
    max_segments: int = 2,
    degree: int = 2,
    breakpoint_guess: Optional[float] = None,
) -> PumpSpeedFit:
    """
    Fit pump speed vs current using piecewise linear fitting (pwlf).

    Automatically detects Imax from breakpoints when 2 segments are used.
    If the second segment is approximately flat (plateau), the breakpoint
    between segments is reported as the detected Imax.

    This method is more sophisticated than the simple fit and can handle
    cases where the pump speed saturates at high currents.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]. Should be filtered (e.g., I >= 300 A).
    pump_speed : np.ndarray
        Pump speed values [rpm].
    max_segments : int, optional
        Maximum number of segments to try (1 or 2). Default is 2.
        The function will try 1 segment first, then 2 if needed.
    degree : int, optional
        Polynomial degree within each segment. Default is 2 (quadratic).
    breakpoint_guess : float, optional
        Initial guess for breakpoint location [A]. If None, pwlf will
        automatically determine the optimal breakpoint.

    Returns
    -------
    PumpSpeedFit
        Fitted parameters with imax_detected=True if breakpoint detection
        succeeded. Includes symbolic equations and breakpoints.

    Raises
    ------
    ImportError
        If pwlf or sympy are not installed.
    ValueError
        If arrays are invalid or max_segments is not 1 or 2.

    Examples
    --------
    >>> import numpy as np
    >>> # Generate data with saturation
    >>> current = np.linspace(1000, 35000, 150)
    >>> pump_speed = np.where(
    ...     current < 28000,
    ...     2840 * (current / 28000)**2 + 1000,
    ...     3840  # Saturated
    ... ) + np.random.normal(0, 10, 150)
    >>> fit = fit_pump_speed_piecewise(current, pump_speed)
    >>> print(f"Detected Imax = {fit.imax:.0f} A")
    >>> print(f"Imax detected: {fit.imax_detected}")

    Notes
    -----
    The piecewise fitting tries segments sequentially:
    1. First tries 1 segment (simple polynomial)
    2. If the fit at the final current is poor (>10 rpm error), tries 2 segments
    3. With 2 segments, the breakpoint is interpreted as Imax

    See Also
    --------
    fit_pump_speed_simple : Simpler method requiring known Imax.
    """
    # Validate inputs
    _validate_array_inputs(current, {"pump_speed": pump_speed})
    if max_segments not in {1, 2}:
        raise ValueError(f"max_segments must be 1 or 2, got {max_segments}")

    # Try to import pwlf
    try:
        import pwlf
    except ImportError as e:
        raise ImportError(
            "pwlf is required for piecewise fitting. Install it with:\n"
            "  pip install pwlf\n"
            "Or install the fitting extras:\n"
            "  pip install python_magnetcooling[fitting]"
        ) from e

    logger.debug(
        f"Fitting pump speed (piecewise method) with max_segments={max_segments}, "
        f"degree={degree}"
    )

    # Try segments sequentially
    best_fit = None
    best_equations = None
    
    for n_segments in range(1, max_segments + 1):
        logger.debug(f"Trying {n_segments} segment(s)...")
        
        # Initialize pwlf model
        my_pwlf = pwlf.PiecewiseLinFit(current, pump_speed, degree=degree)
        
        # Fit with or without breakpoint guess
        if breakpoint_guess is not None and n_segments == 2:
            logger.debug(f"Using breakpoint guess: {breakpoint_guess} A")
            my_pwlf.fit_guess([breakpoint_guess])
        else:
            my_pwlf.fit(n_segments)
        
        # Extract symbolic equations
        try:
            equations = _extract_piecewise_equations(my_pwlf)
        except ImportError:
            logger.warning("sympy not available, skipping equation extraction")
            equations = None
        
        # Check fit quality at the final point
        if equations is not None:
            from sympy import Symbol
            x = Symbol("x")
            final_current = current[-1]
            final_speed = pump_speed[-1]
            predicted_speed = float(equations[0].evalf(subs={x: final_current}))
            error = abs(predicted_speed - final_speed)
            
            logger.debug(
                f"Fit quality check: error at I={final_current:.0f} A is {error:.1f} rpm"
            )
            
            # If error is acceptable or this is the last attempt, use this fit
            if error <= 10 or n_segments == max_segments:
                best_fit = my_pwlf
                best_equations = equations
                break
        else:
            # No equations available, use this fit
            best_fit = my_pwlf
            best_equations = None
            break
    
    if best_fit is None:
        raise RuntimeError("Piecewise fitting failed for all segment configurations")
    
    # Extract parameters
    n_segments = best_fit.n_segments
    breakpoints_list = list(best_fit.fit_breaks)
    
    # Determine Imax
    if n_segments == 2:
        # Breakpoint between segments is the detected Imax
        detected_imax = breakpoints_list[1]
        imax_detected = True
        logger.info(f"Detected Imax from breakpoint: {detected_imax:.0f} A")
    else:
        # Use the maximum current value as Imax
        detected_imax = float(current.max())
        imax_detected = False
        logger.info(f"Using data maximum as Imax: {detected_imax:.0f} A")
    
    # Evaluate Vp0 and Vpmax from the equation
    if best_equations is not None:
        from sympy import Symbol
        x = Symbol("x")
        vp0 = float(best_equations[0].evalf(subs={x: 0}))
        vpmax = float(best_equations[0].evalf(subs={x: detected_imax}))
    else:
        # Fallback: evaluate using pwlf predict
        vp0 = float(best_fit.predict(0))
        vpmax = float(best_fit.predict(detected_imax))
    
    logger.info(
        f"Pump speed fit complete (piecewise): Vpmax={vpmax:.2f} rpm, "
        f"Vp0={vp0:.2f} rpm, Imax={detected_imax:.0f} A (detected={imax_detected}), "
        f"segments={n_segments}"
    )
    
    return PumpSpeedFit(
        vpmax=vpmax,
        vp0=vp0,
        imax=detected_imax,
        imax_detected=imax_detected,
        fit_result=None,  # pwlf doesn't provide scipy-style covariance
        breakpoints=breakpoints_list,
        equations=best_equations,
    )


# =============================================================================
# Flow Rate and Pressure Fitting Functions
# =============================================================================


def fit_flow_rate(
    current: np.ndarray,
    flow_rate: np.ndarray,
    pump_fit: PumpSpeedFit,
) -> FitResult:
    """
    Fit flow rate vs current.

    Model: F(I) = F0 + Fmax · Vp(I) / (Vpmax + Vp0)

    where Vp(I) is the pump speed model from pump_fit.

    This function fits the flow rate characteristic curve based on the
    already-fitted pump speed parameters. The flow rate is assumed to be
    linearly related to the normalized pump speed.

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    flow_rate : np.ndarray
        Flow rate values [l/s].
    pump_fit : PumpSpeedFit
        Previously fitted pump speed parameters.

    Returns
    -------
    FitResult
        Fit statistics with parameters = [F0, Fmax].

    Raises
    ------
    ValueError
        If arrays are invalid or have mismatched lengths.
    RuntimeError
        If the curve fitting fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> from python_magnetcooling.fitting import fit_pump_speed_simple, fit_flow_rate
    >>> 
    >>> # First fit pump speed
    >>> current = np.linspace(1000, 28000, 100)
    >>> pump_speed = 2840 * (current / 28000)**2 + 1000
    >>> pump_fit = fit_pump_speed_simple(current, pump_speed, imax=28000)
    >>> 
    >>> # Then fit flow rate
    >>> flow = 140 * pump_speed / (2840 + 1000)
    >>> flow_fit = fit_flow_rate(current, flow, pump_fit)
    >>> print(f"F0 = {flow_fit.parameters[0]:.2f} l/s")
    >>> print(f"Fmax = {flow_fit.parameters[1]:.2f} l/s")
    >>> print(f"R² = {flow_fit.r_squared:.6f}")

    See Also
    --------
    fit_pump_speed_simple : Fit pump speed first.
    fit_pressure : Fit pressure curve.
    """
    # Validate inputs
    _validate_array_inputs(current, {"flow_rate": flow_rate})

    logger.debug("Fitting flow rate curve")

    # Get pump parameters
    vpmax = pump_fit.vpmax
    vp0 = pump_fit.vp0
    imax = pump_fit.imax

    # Define pump speed function
    def vpump_func(x: np.ndarray) -> np.ndarray:
        """Vp(I) = Vpmax·(I/Imax)² + Vp0"""
        return vpmax * (x / imax) ** 2 + vp0

    # Define flow rate model
    def flow_func(x: np.ndarray, f0: float, fmax: float) -> np.ndarray:
        """F(I) = F0 + Fmax · Vp(I) / (Vpmax + Vp0)"""
        vp = vpump_func(x)
        return f0 + fmax * vp / (vpmax + vp0)

    try:
        # Perform curve fitting
        params, params_covariance = optimize.curve_fit(
            flow_func, current, flow_rate
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Flow rate curve fitting failed to converge: {e}"
        ) from e

    f0, fmax = params

    # Compute fit quality
    y_fit = flow_func(current, f0, fmax)
    _, standard_errors, r_squared, residuals = _compute_fit_statistics(
        flow_rate, y_fit, params_covariance
    )

    logger.info(
        f"Flow rate fit complete: F0={f0:.2f} l/s, Fmax={fmax:.2f} l/s, "
        f"R²={r_squared:.6f}"
    )

    return FitResult(
        parameters=params,
        standard_errors=standard_errors,
        r_squared=r_squared,
        residuals=residuals,
    )


def fit_pressure(
    current: np.ndarray,
    pressure: np.ndarray,
    pump_fit: PumpSpeedFit,
) -> FitResult:
    """
    Fit inlet pressure vs current.

    Model: P(I) = Pmin + Pmax · [Vp(I) / (Vpmax + Vp0)]²

    where Vp(I) is the pump speed model from pump_fit.

    This function fits the pressure characteristic curve based on the
    already-fitted pump speed parameters. The pressure is assumed to be
    quadratically related to the normalized pump speed.

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    pressure : np.ndarray
        Inlet pressure values [bar].
    pump_fit : PumpSpeedFit
        Previously fitted pump speed parameters.

    Returns
    -------
    FitResult
        Fit statistics with parameters = [Pmin, Pmax].

    Raises
    ------
    ValueError
        If arrays are invalid or have mismatched lengths.
    RuntimeError
        If the curve fitting fails to converge.

    Examples
    --------
    >>> import numpy as np
    >>> from python_magnetcooling.fitting import fit_pump_speed_simple, fit_pressure
    >>> 
    >>> # First fit pump speed
    >>> current = np.linspace(1000, 28000, 100)
    >>> pump_speed = 2840 * (current / 28000)**2 + 1000
    >>> pump_fit = fit_pump_speed_simple(current, pump_speed, imax=28000)
    >>> 
    >>> # Then fit pressure
    >>> vp_ratio = pump_speed / (2840 + 1000)
    >>> pressure = 4 + 22 * vp_ratio**2
    >>> pressure_fit = fit_pressure(current, pressure, pump_fit)
    >>> print(f"Pmin = {pressure_fit.parameters[0]:.2f} bar")
    >>> print(f"Pmax = {pressure_fit.parameters[1]:.2f} bar")
    >>> print(f"R² = {pressure_fit.r_squared:.6f}")

    See Also
    --------
    fit_pump_speed_simple : Fit pump speed first.
    fit_flow_rate : Fit flow rate curve.
    """
    # Validate inputs
    _validate_array_inputs(current, {"pressure": pressure})

    logger.debug("Fitting pressure curve")

    # Get pump parameters
    vpmax = pump_fit.vpmax
    vp0 = pump_fit.vp0
    imax = pump_fit.imax

    # Define pump speed function
    def vpump_func(x: np.ndarray) -> np.ndarray:
        """Vp(I) = Vpmax·(I/Imax)² + Vp0"""
        return vpmax * (x / imax) ** 2 + vp0

    # Define pressure model
    def pressure_func(x: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
        """P(I) = Pmin + Pmax · [Vp(I) / (Vpmax + Vp0)]²"""
        vp = vpump_func(x)
        return pmin + pmax * (vp / (vpmax + vp0)) ** 2

    try:
        # Perform curve fitting
        params, params_covariance = optimize.curve_fit(
            pressure_func, current, pressure
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Pressure curve fitting failed to converge: {e}"
        ) from e

    pmin, pmax = params

    # Compute fit quality
    y_fit = pressure_func(current, pmin, pmax)
    _, standard_errors, r_squared, residuals = _compute_fit_statistics(
        pressure, y_fit, params_covariance
    )

    logger.info(
        f"Pressure fit complete: Pmin={pmin:.2f} bar, Pmax={pmax:.2f} bar, "
        f"R²={r_squared:.6f}"
    )

    return FitResult(
        parameters=params,
        standard_errors=standard_errors,
        r_squared=r_squared,
        residuals=residuals,
    )


def compute_back_pressure_stats(
    back_pressure: np.ndarray,
) -> tuple[float, float]:
    """
    Compute mean and standard deviation of back pressure.

    Back pressure is typically constant or slowly varying, so we compute
    statistics rather than fitting a curve.

    Parameters
    ----------
    back_pressure : np.ndarray
        Back pressure measurements [bar].

    Returns
    -------
    mean : float
        Mean back pressure [bar].
    std : float
        Standard deviation of back pressure [bar].

    Raises
    ------
    ValueError
        If back_pressure array is invalid or empty.

    Examples
    --------
    >>> import numpy as np
    >>> back_pressure = np.array([4.0, 4.1, 3.9, 4.0, 4.1])
    >>> mean, std = compute_back_pressure_stats(back_pressure)
    >>> print(f"Back pressure: {mean:.2f} ± {std:.2f} bar")
    Back pressure: 4.02 ± 0.08 bar

    See Also
    --------
    fit_hydraulic_system : Complete fitting pipeline.
    """
    if not isinstance(back_pressure, np.ndarray):
        raise ValueError(
            f"back_pressure must be a numpy array, got {type(back_pressure)}"
        )

    if len(back_pressure) == 0:
        raise ValueError("back_pressure array is empty")

    if np.any(~np.isfinite(back_pressure)):
        raise ValueError("back_pressure array contains NaN or Inf values")

    mean = float(np.mean(back_pressure))
    std = float(np.std(back_pressure))

    logger.debug(f"Back pressure statistics: mean={mean:.2f} bar, std={std:.4f} bar")

    return mean, std


# =============================================================================
# Orchestration Function
# =============================================================================


def fit_hydraulic_system(
    current: np.ndarray,
    pump_speed: np.ndarray,
    flow_rate: np.ndarray,
    pressure: np.ndarray,
    back_pressure: np.ndarray,
    imax: Optional[float] = None,
    method: str = "simple",
    current_threshold: float = 300.0,
) -> tuple[PumpSpeedFit, FlowPressureFit]:
    """
    Complete fitting pipeline: pump speed → flow → pressure → back pressure.

    This is the main entry point for fitting all hydraulic curves from
    experimental arrays. It orchestrates the entire fitting workflow:
    1. Filter data by current threshold
    2. Fit pump speed curve
    3. Fit flow rate curve (using pump parameters)
    4. Fit pressure curve (using pump parameters)
    5. Compute back pressure statistics
    6. Return typed result objects

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    pump_speed : np.ndarray
        Pump speed values [rpm].
    flow_rate : np.ndarray
        Flow rate values [l/s].
    pressure : np.ndarray
        Inlet pressure values [bar].
    back_pressure : np.ndarray
        Back pressure values [bar].
    imax : float, optional
        Maximum current [A]. Required if method="simple".
        Auto-detected if method="piecewise" and not provided.
    method : str, optional
        Fitting method: "simple" (scipy quadratic) or "piecewise" (pwlf with
        breakpoint detection). Default is "simple".
    current_threshold : float, optional
        Minimum current threshold [A] for filtering data. Default is 300 A.
        Data points with current < threshold are excluded from fitting.

    Returns
    -------
    pump_fit : PumpSpeedFit
        Pump speed fit results.
    flow_pressure_fit : FlowPressureFit
        Combined flow rate and pressure fit results.

    Raises
    ------
    ValueError
        If arrays are invalid, have mismatched lengths, or method is invalid.
    RuntimeError
        If any curve fitting fails to converge.
    ImportError
        If method="piecewise" but pwlf is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> from python_magnetcooling.fitting import fit_hydraulic_system
    >>> 
    >>> # Generate synthetic data
    >>> np.random.seed(42)
    >>> current = np.linspace(500, 28000, 200)
    >>> pump_speed = 2840 * (current / 28000)**2 + 1000 + np.random.normal(0, 20, 200)
    >>> vp_ratio = pump_speed / (2840 + 1000)
    >>> flow = 140 * vp_ratio + np.random.normal(0, 1, 200)
    >>> pressure = 4 + 22 * vp_ratio**2 + np.random.normal(0, 0.5, 200)
    >>> back_pressure = np.full(200, 4.0) + np.random.normal(0, 0.1, 200)
    >>> 
    >>> # Fit all curves at once
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(
    ...     current, pump_speed, flow, pressure, back_pressure,
    ...     imax=28000, method="simple", current_threshold=300
    ... )
    >>> 
    >>> print(f"Pump: Vpmax={pump_fit.vpmax:.1f} rpm, Vp0={pump_fit.vp0:.1f} rpm")
    >>> print(f"Flow: F0={flow_pressure_fit.f0:.1f} l/s, Fmax={flow_pressure_fit.fmax:.1f} l/s")
    >>> print(f"Pressure: Pmin={flow_pressure_fit.pmin:.1f} bar, Pmax={flow_pressure_fit.pmax:.1f} bar")
    >>> print(f"Back pressure: {flow_pressure_fit.back_pressure:.1f} ± {flow_pressure_fit.back_pressure_std:.2f} bar")

    See Also
    --------
    fit_pump_speed_simple : Simple pump speed fitting.
    fit_pump_speed_piecewise : Advanced pump speed fitting with Imax detection.
    build_waterflow : Create WaterFlow object from fit results.
    """
    # Validate method selection
    _validate_method(method)
    _validate_imax(imax, method)

    logger.info(
        f"Starting hydraulic system fitting (method={method}, "
        f"threshold={current_threshold} A)"
    )

    # Validate all input arrays (before filtering)
    _validate_array_inputs(
        current,
        {
            "pump_speed": pump_speed,
            "flow_rate": flow_rate,
            "pressure": pressure,
            "back_pressure": back_pressure,
        },
    )

    # Filter data by current threshold
    current_filtered, pump_speed_filtered, flow_rate_filtered, pressure_filtered, back_pressure_filtered = _filter_by_threshold(
        current, current_threshold, pump_speed, flow_rate, pressure, back_pressure
    )

    logger.info(f"Data filtered: {len(current)} → {len(current_filtered)} points")

    # Step 1: Fit pump speed curve
    if method == "simple":
        pump_fit = fit_pump_speed_simple(
            current_filtered, pump_speed_filtered, imax=imax
        )
    elif method == "piecewise":
        pump_fit = fit_pump_speed_piecewise(
            current_filtered, pump_speed_filtered, max_segments=2
        )
    else:
        # Should never reach here due to _validate_method
        raise ValueError(f"Invalid method: {method}")

    # Step 2: Fit flow rate curve
    flow_fit = fit_flow_rate(current_filtered, flow_rate_filtered, pump_fit)

    # Step 3: Fit pressure curve
    pressure_fit = fit_pressure(current_filtered, pressure_filtered, pump_fit)

    # Step 4: Compute back pressure statistics
    bp_mean, bp_std = compute_back_pressure_stats(back_pressure_filtered)

    # Build combined result
    flow_pressure_fit = FlowPressureFit(
        f0=flow_fit.parameters[0],
        fmax=flow_fit.parameters[1],
        pmin=pressure_fit.parameters[0],
        pmax=pressure_fit.parameters[1],
        back_pressure=bp_mean,
        back_pressure_std=bp_std,
        flow_fit=flow_fit,
        pressure_fit=pressure_fit,
    )

    logger.info(
        f"Hydraulic system fitting complete. "
        f"Pump R²={pump_fit.fit_result.r_squared if pump_fit.fit_result else 'N/A':.4f}, "
        f"Flow R²={flow_fit.r_squared:.4f}, "
        f"Pressure R²={pressure_fit.r_squared:.4f}"
    )

    return pump_fit, flow_pressure_fit


# =============================================================================
# Hysteresis Fitting
# =============================================================================


def fit_hysteresis_parameters(
    power: np.ndarray,
    flow_rate: np.ndarray,
    n_levels: Optional[int] = None,
    clean_outliers: bool = True,
    outlier_percentile: float = 25,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    verbose: bool = False,
) -> HysteresisFit:
    """
    Fit multi-level hysteresis parameters from power/flow time-series data.

    Estimates hysteresis model parameters by analyzing transitions in flow rate
    as power changes. The model captures different flow rates for ascending vs
    descending power, which is common in magnet cooling systems where operators
    adjust flow rates conservatively.

    Parameters
    ----------
    power : np.ndarray
        Magnet power values [MW], sequential time-series data.
    flow_rate : np.ndarray
        Secondary cooling loop flow rate values [m³/h], same length as power.
        (Formerly "debitbrut" - use "flow_secondary" in new CSV files)
    n_levels : int, optional
        Number of discrete levels to cluster flow rates into.
        If None, auto-detects distinct levels from data.
        Recommended: 3-5 levels for typical systems.
    clean_outliers : bool, default=True
        If True, remove outliers from low-power region before fitting.
        Recommended to improve fit quality.
    outlier_percentile : float, default=25
        Define "low power" as values below this percentile (0-100).
        Only used if clean_outliers=True.
    outlier_method : str, default="iqr"
        Outlier removal method: 'iqr', 'zscore', or 'both_dims'.
        Only used if clean_outliers=True.
    outlier_threshold : float, default=1.5
        Threshold for outlier detection (IQR multiplier or z-score).
        Only used if clean_outliers=True.
    verbose : bool, default=False
        If True, print diagnostic information during fitting.

    Returns
    -------
    HysteresisFit
        Fitted hysteresis parameters including thresholds, low_values,
        high_values, and diagnostics.

    Raises
    ------
    ValueError
        If input arrays are invalid, empty, or have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from python_magnetcooling.fitting import fit_hysteresis_parameters
    >>> 
    >>> # Load magnet run data
    >>> df = pd.read_csv("magnet_data.csv")
    >>> power = df["Pmagnet"].values  # MW
    >>> flow = df["flow_secondary"].values  # m³/h (secondary cooling loop)
    >>> 
    >>> # Fit hysteresis model with 3 levels
    >>> hyst_fit = fit_hysteresis_parameters(
    ...     power, flow,
    ...     n_levels=3,
    ...     clean_outliers=True,
    ...     verbose=True
    ... )
    >>> 
    >>> print(f"Thresholds: {hyst_fit.thresholds}")
    >>> print(f"Low values: {hyst_fit.low_values}")
    >>> print(f"High values: {hyst_fit.high_values}")
    >>> print(f"Valid levels: {hyst_fit.diagnostics['n_valid_levels']}")

    See Also
    --------
    build_waterflow_with_hysteresis : Create WaterFlow with hysteresis support.
    python_magnetcooling.hysteresis.estimate_hysteresis_parameters : Underlying function.
    python_magnetcooling.hysteresis.multi_level_hysteresis : Apply the model.

    Notes
    -----
    The function wraps the core hysteresis estimation from the hysteresis module
    and provides a fitting interface consistent with other fitting functions in
    this module.

    The data must be sequential (time-ordered) for transition detection to work.
    Randomly sampled or re-ordered data will produce invalid results.
    """
    # Validate inputs
    if not isinstance(power, np.ndarray):
        raise ValueError(f"power must be a numpy array, got {type(power)}")
    if not isinstance(flow_rate, np.ndarray):
        raise ValueError(f"flow_rate must be a numpy array, got {type(flow_rate)}")

    if len(power) == 0 or len(flow_rate) == 0:
        raise ValueError("power and flow_rate arrays cannot be empty")

    if len(power) != len(flow_rate):
        raise ValueError(
            f"power and flow_rate must have same length. "
            f"Got {len(power)} and {len(flow_rate)}"
        )

    if np.any(~np.isfinite(power)) or np.any(~np.isfinite(flow_rate)):
        raise ValueError("power and flow_rate arrays must not contain NaN or Inf")

    if len(power) < 10:
        raise ValueError(
            f"Insufficient data points for hysteresis fitting. "
            f"Need at least 10, got {len(power)}"
        )

    logger.info(
        f"Fitting hysteresis parameters: {len(power)} points, "
        f"n_levels={n_levels}, clean_outliers={clean_outliers}"
    )

    # Create DataFrame for processing
    import pandas as pd
    df = pd.DataFrame({"Pmagnet": power, "flow_secondary": flow_rate})

    # Clean outliers if requested
    if clean_outliers:
        from .hysteresis import remove_low_x_outliers

        logger.debug(
            f"Removing outliers: percentile={outlier_percentile}, "
            f"method={outlier_method}, threshold={outlier_threshold}"
        )

        df_clean = remove_low_x_outliers(
            df,
            x_col="Pmagnet",
            y_col="flow_secondary",
            x_percentile=outlier_percentile,
            method=outlier_method,
            threshold=outlier_threshold,
            verbose=verbose,
        )

        n_removed = len(df) - len(df_clean)
        logger.info(f"Removed {n_removed} outliers ({100 * n_removed / len(df):.1f}%)")
    else:
        df_clean = df

    # Estimate hysteresis parameters
    from .hysteresis import estimate_hysteresis_parameters

    result = estimate_hysteresis_parameters(
        df_clean, x_col="Pmagnet", y_col="flow_secondary", n_levels=n_levels, verbose=verbose
    )

    # Extract results
    thresholds = result["thresholds"]
    low_values = result["low_values"]
    high_values = result["high_values"]
    diagnostics = result["diagnostics"]

    if not thresholds:
        raise ValueError(
            "No valid hysteresis levels detected. Try adjusting n_levels or "
            "check that your data contains clear flow rate transitions."
        )

    logger.info(
        f"Hysteresis fit complete: {len(thresholds)} valid levels, "
        f"{diagnostics['n_transitions']} transitions analyzed"
    )

    # Create and return result
    hyst_fit = HysteresisFit(
        thresholds=thresholds,
        low_values=low_values,
        high_values=high_values,
        diagnostics=diagnostics,
    )

    return hyst_fit


# =============================================================================
# WaterFlow Construction
# =============================================================================


def build_waterflow(
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
) -> "WaterFlow":
    """
    Construct a WaterFlow object from fitted parameters.

    This function replaces manual dict construction and provides a typed
    path from fit results to WaterFlow objects. It maps the fitted hydraulic
    parameters to WaterFlow constructor arguments.

    Parameters
    ----------
    pump_fit : PumpSpeedFit
        Pump speed fit results.
    flow_pressure_fit : FlowPressureFit
        Flow rate and pressure fit results.

    Returns
    -------
    WaterFlow
        Configured WaterFlow instance ready for hydraulic calculations.

    Examples
    --------
    >>> import numpy as np
    >>> from python_magnetcooling.fitting import fit_hydraulic_system, build_waterflow
    >>> 
    >>> # Generate and fit synthetic data
    >>> current = np.linspace(1000, 28000, 100)
    >>> pump_speed = 2840 * (current / 28000)**2 + 1000
    >>> vp_ratio = pump_speed / (2840 + 1000)
    >>> flow = 140 * vp_ratio
    >>> pressure = 4 + 22 * vp_ratio**2
    >>> back_pressure = np.full(100, 4.0)
    >>> 
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(
    ...     current, pump_speed, flow, pressure, back_pressure,
    ...     imax=28000, method="simple"
    ... )
    >>> 
    >>> # Build WaterFlow object
    >>> waterflow = build_waterflow(pump_fit, flow_pressure_fit)
    >>> print(f"Max flow: {waterflow.flow_max:.1f} l/s")
    >>> print(f"Max pressure: {waterflow.pressure_max:.1f} bar")
    >>> 
    >>> # Use it for calculations
    >>> flow_at_20kA = waterflow.flow_rate(20000)
    >>> print(f"Flow at 20 kA: {flow_at_20kA:.2f} m³/s")

    See Also
    --------
    fit_hydraulic_system : Fit all curves from experimental data.
    waterflow_factory.from_fits : Alternative factory method.
    """
    from .waterflow import WaterFlow

    logger.debug("Building WaterFlow object from fit results")

    waterflow = WaterFlow(
        pump_speed_min=pump_fit.vp0,
        pump_speed_max=pump_fit.vpmax,
        flow_min=flow_pressure_fit.f0,
        flow_max=flow_pressure_fit.fmax,
        pressure_max=flow_pressure_fit.pmax,
        pressure_min=flow_pressure_fit.pmin,
        pressure_back=flow_pressure_fit.back_pressure,
        current_max=pump_fit.imax,
    )

    logger.info(
        f"WaterFlow object created: "
        f"pump {waterflow.pump_speed_min}-{waterflow.pump_speed_max} rpm, "
        f"flow {waterflow.flow_min}-{waterflow.flow_max} l/s, "
        f"pressure {waterflow.pressure_min}-{waterflow.pressure_max} bar, "
        f"Imax {waterflow.current_max} A"
    )

    return waterflow


def build_waterflow_with_hysteresis(
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
    hyst_fit: HysteresisFit,
) -> "WaterFlow":
    """
    Construct a WaterFlow object with hysteresis parameters from fitted data.

    This function combines standard hydraulic parameters with hysteresis model
    parameters to create a WaterFlow object that supports both standard flow
    calculations and power-dependent flow with hysteresis for secondary cooling loop flow.

    Parameters
    ----------
    pump_fit : PumpSpeedFit
        Pump speed fit results.
    flow_pressure_fit : FlowPressureFit
        Flow rate and pressure fit results.
    hyst_fit : HysteresisFit
        Hysteresis parameter fit results.

    Returns
    -------
    WaterFlow
        Configured WaterFlow instance with hysteresis support.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from python_magnetcooling.fitting import (
    ...     fit_hydraulic_system,
    ...     fit_hysteresis_parameters,
    ...     build_waterflow_with_hysteresis
    ... )
    >>> 
    >>> # Fit hydraulic system
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(
    ...     current, pump_speed, flow, pressure, back_pressure,
    ...     imax=28000, method="simple"
    ... )
    >>> 
    >>> # Fit hysteresis from power/flow data
    >>> hyst_fit = fit_hysteresis_parameters(power, flow_rate, n_levels=3)
    >>> 
    >>> # Build complete WaterFlow object
    >>> waterflow = build_waterflow_with_hysteresis(
    ...     pump_fit, flow_pressure_fit, hyst_fit
    ... )
    >>> 
    >>> # Use standard methods
    >>> flow_at_20kA = waterflow.flow_rate(20000)
    >>> 
    >>> # Use hysteresis method
    >>> flow_at_10MW = waterflow.debitbrut(10.0)  # MW -> returns secondary flow in m³/h

    See Also
    --------
    build_waterflow : Build WaterFlow without hysteresis.
    fit_hysteresis_parameters : Fit hysteresis parameters.
    fit_hydraulic_system : Fit standard hydraulic parameters.
    """
    from .waterflow import WaterFlow

    logger.debug("Building WaterFlow object with hysteresis from fit results")

    waterflow = WaterFlow(
        pump_speed_min=pump_fit.vp0,
        pump_speed_max=pump_fit.vpmax,
        flow_min=flow_pressure_fit.f0,
        flow_max=flow_pressure_fit.fmax,
        pressure_max=flow_pressure_fit.pmax,
        pressure_min=flow_pressure_fit.pmin,
        pressure_back=flow_pressure_fit.back_pressure,
        current_max=pump_fit.imax,
        hysteresis_thresholds=hyst_fit.thresholds,
        hysteresis_low_values=hyst_fit.low_values,
        hysteresis_high_values=hyst_fit.high_values,
    )

    logger.info(
        f"WaterFlow object created with hysteresis: "
        f"pump {waterflow.pump_speed_min}-{waterflow.pump_speed_max} rpm, "
        f"flow {waterflow.flow_min}-{waterflow.flow_max} l/s, "
        f"pressure {waterflow.pressure_min}-{waterflow.pressure_max} bar, "
        f"Imax {waterflow.current_max} A, "
        f"hysteresis levels: {len(hyst_fit.thresholds)}"
    )

    return waterflow


# ============================================================================
# Fit Quality Metrics
# ============================================================================


def _compute_metrics_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> FitMetrics:
    """
    Compute fit quality metrics from true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        True/measured values
    y_pred : np.ndarray
        Predicted/fitted values

    Returns
    -------
    FitMetrics
        Comprehensive fit quality metrics
    """
    residuals = y_true - y_pred
    n_points = len(y_true)

    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    # MAE
    mae = np.mean(np.abs(residuals))

    # Max error
    max_error = np.max(np.abs(residuals))

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # MAPE (only if no zeros in true values)
    if np.all(y_true != 0):
        mape = np.mean(np.abs(residuals / y_true)) * 100
    else:
        mape = None

    return FitMetrics(
        rmse=float(rmse),
        mae=float(mae),
        max_error=float(max_error),
        r_squared=float(r_squared),
        mape=float(mape) if mape is not None else None,
        n_points=int(n_points),
    )


def compute_pump_fit_metrics(
    current: np.ndarray, pump_speed: np.ndarray, pump_fit: PumpSpeedFit
) -> FitMetrics:
    """
    Compute fit quality metrics for pump speed model.

    Evaluates how well the fitted quadratic model matches the measured
    pump speed data. Use these metrics to decide if a refit is necessary.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]
    pump_speed : np.ndarray
        Measured pump speed values [rpm]
    pump_fit : PumpSpeedFit
        Fitted pump parameters from fit_pump_speed()

    Returns
    -------
    FitMetrics
        Fit quality metrics including RMSE, MAE, R², etc.

    Examples
    --------
    >>> pump_fit, _ = fit_hydraulic_system(current, pump_speed, flow, pressure, back_p)
    >>> metrics = compute_pump_fit_metrics(current, pump_speed, pump_fit)
    >>> print(metrics)
    Fit Quality Metrics:
      RMSE:      12.5432
      MAE:       8.3421
      Max Error: 35.2134
      R²:        0.985421
      MAPE:      2.13%
      N points:  100
    >>> if not metrics.is_good_fit(rmse_threshold=20):
    ...     print("Refit recommended")

    Notes
    -----
    **Typical acceptable values for pump fits:**
    
    - RMSE: <20 rpm for standard pumps
    - R²: >0.95 is excellent, >0.90 is acceptable
    - MAE: <15 rpm indicates good fit
    
    Consider refit if R² < 0.90 or if residuals show systematic patterns.
    """
    # Reconstruct fitted pump speed
    vp_fit = pump_fit.vpmax * (current / pump_fit.imax) ** 2 + pump_fit.vp0

    return _compute_metrics_from_arrays(pump_speed, vp_fit)


def compute_flow_fit_metrics(
    current: np.ndarray,
    flow_rate: np.ndarray,
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
) -> FitMetrics:
    """
    Compute fit quality metrics for flow rate model.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]
    flow_rate : np.ndarray
        Measured flow rate values [l/s]
    pump_fit : PumpSpeedFit
        Fitted pump parameters
    flow_pressure_fit : FlowPressureFit
        Fitted flow/pressure parameters

    Returns
    -------
    FitMetrics
        Fit quality metrics

    Examples
    --------
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(...)
    >>> metrics = compute_flow_fit_metrics(current, flow_rate, pump_fit,
    ...                                     flow_pressure_fit)
    >>> print(f"Flow fit R² = {metrics.r_squared:.4f}")

    Notes
    -----
    **Typical acceptable values:**
    
    - RMSE: <2 l/s for most systems
    - R²: >0.90 is good
    - Consider refit if MAPE > 10%
    """
    # Reconstruct fitted flow
    vp_fit = pump_fit.vpmax * (current / pump_fit.imax) ** 2 + pump_fit.vp0
    flow_fit = flow_pressure_fit.f0 + flow_pressure_fit.fmax * vp_fit / (
        pump_fit.vpmax + pump_fit.vp0
    )

    return _compute_metrics_from_arrays(flow_rate, flow_fit)


def compute_pressure_fit_metrics(
    current: np.ndarray,
    pressure: np.ndarray,
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
) -> FitMetrics:
    """
    Compute fit quality metrics for pressure model.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]
    pressure : np.ndarray
        Measured pressure values [bar]
    pump_fit : PumpSpeedFit
        Fitted pump parameters
    flow_pressure_fit : FlowPressureFit
        Fitted flow/pressure parameters

    Returns
    -------
    FitMetrics
        Fit quality metrics

    Examples
    --------
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(...)
    >>> metrics = compute_pressure_fit_metrics(current, pressure, pump_fit,
    ...                                         flow_pressure_fit)
    >>> if metrics.rmse > 1.0:
    ...     print("Pressure fit has high error, consider refit")

    Notes
    -----
    **Typical acceptable values:**
    
    - RMSE: <0.5 bar is excellent, <1.0 bar is acceptable
    - R²: >0.95 is good
    - Check that fitted Pmin and Pmax are physically reasonable
    """
    # Reconstruct fitted pressure
    vp_fit = pump_fit.vpmax * (current / pump_fit.imax) ** 2 + pump_fit.vp0
    pressure_fit = flow_pressure_fit.pmin + flow_pressure_fit.pmax * (
        vp_fit / (pump_fit.vpmax + pump_fit.vp0)
    ) ** 2

    return _compute_metrics_from_arrays(pressure, pressure_fit)


def compute_hysteresis_fit_metrics(
    x: np.ndarray, y: np.ndarray, hyst_fit: HysteresisFit
) -> FitMetrics:
    """
    Compute fit quality metrics for hysteresis model.

    Evaluates how well the fitted hysteresis model reproduces the measured
    power-flow (or similar) relationship. Use to assess model quality and
    decide if parameter re-estimation is needed.

    Parameters
    ----------
    x : np.ndarray
        Input values (e.g., power in MW)
    y : np.ndarray
        Output values (e.g., flow rate in m³/h)
    hyst_fit : HysteresisFit
        Fitted hysteresis parameters from fit_hysteresis_parameters()

    Returns
    -------
    FitMetrics
        Fit quality metrics

    Examples
    --------
    >>> hyst_fit = fit_hysteresis_parameters(power, flow_rate, n_levels=3)
    >>> metrics = compute_hysteresis_fit_metrics(power, flow_rate, hyst_fit)
    >>> print(metrics)
    >>> if not metrics.is_good_fit(r_squared_threshold=0.85):
    ...     print("Hysteresis model quality is poor, consider:")
    ...     print("  - Different n_levels parameter")
    ...     print("  - Outlier removal before fitting")
    ...     print("  - Check for data quality issues")

    Notes
    -----
    **Interpretation for hysteresis models:**
    
    - R² > 0.90: Excellent fit, clear hysteresis behavior
    - R² 0.80-0.90: Good fit, hysteresis present but with noise
    - R² < 0.80: Poor fit, consider:
      * Different number of levels
      * Better outlier removal
      * Data may not exhibit clear hysteresis
    
    **When to refit:**
    
    - New data collected that shows different behavior
    - Systematic deviations in specific power ranges
    - Control system parameters changed
    - RMSE > 10% of typical flow rate range
    """
    from .hysteresis import multi_level_hysteresis

    # Reconstruct fitted values
    y_fit = multi_level_hysteresis(
        x, hyst_fit.thresholds, hyst_fit.low_values, hyst_fit.high_values
    )

    return _compute_metrics_from_arrays(y, y_fit)


def compute_all_hydraulic_metrics(
    current: np.ndarray,
    pump_speed: np.ndarray,
    flow_rate: np.ndarray,
    pressure: np.ndarray,
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
) -> dict[str, FitMetrics]:
    """
    Compute fit quality metrics for all hydraulic fits at once.

    Convenient function to evaluate pump, flow, and pressure fits together.
    Returns a dictionary with metrics for each component.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]
    pump_speed : np.ndarray
        Measured pump speed values [rpm]
    flow_rate : np.ndarray
        Measured flow rate values [l/s]
    pressure : np.ndarray
        Measured pressure values [bar]
    pump_fit : PumpSpeedFit
        Fitted pump parameters
    flow_pressure_fit : FlowPressureFit
        Fitted flow/pressure parameters

    Returns
    -------
    dict[str, FitMetrics]
        Dictionary with keys 'pump', 'flow', 'pressure' containing
        respective FitMetrics objects

    Examples
    --------
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(
    ...     current, pump_speed, flow, pressure, back_p
    ... )
    >>> metrics = compute_all_hydraulic_metrics(
    ...     current, pump_speed, flow, pressure, pump_fit, flow_pressure_fit
    ... )
    >>> 
    >>> print("=== Fit Quality Summary ===")
    >>> for name, m in metrics.items():
    ...     print(f"\\n{name.upper()}:")
    ...     print(f"  RMSE = {m.rmse:.3f}, R² = {m.r_squared:.4f}")
    ...     if not m.is_good_fit():
    ...         print(f"  ⚠ WARNING: {name} fit quality is low!")
    >>> 
    >>> # Check if any fit needs improvement
    >>> poor_fits = [name for name, m in metrics.items() 
    ...              if not m.is_good_fit(r_squared_threshold=0.90)]
    >>> if poor_fits:
    ...     print(f"\\nConsider refitting: {', '.join(poor_fits)}")

    Notes
    -----
    This is the recommended way to evaluate overall hydraulic system fit quality.
    It provides a comprehensive view of all model components at once.
    """
    return {
        "pump": compute_pump_fit_metrics(current, pump_speed, pump_fit),
        "flow": compute_flow_fit_metrics(current, flow_rate, pump_fit, flow_pressure_fit),
        "pressure": compute_pressure_fit_metrics(
            current, pressure, pump_fit, flow_pressure_fit
        ),
    }


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_pump_fit(
    current: np.ndarray,
    pump_speed: np.ndarray,
    pump_fit: PumpSpeedFit,
    xlabel: str = "Current (A)",
    ylabel: str = "Pump Speed (rpm)",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Plot raw pump speed data and fitted curve.

    Parameters
    ----------
    current : np.ndarray
        Current values (A)
    pump_speed : np.ndarray
        Measured pump speed values (rpm)
    pump_fit : PumpFit
        Fitted pump parameters from fit_pump_speed()
    xlabel : str, default="Current (A)"
        Label for x-axis
    ylabel : str, default="Pump Speed (rpm)"
        Label for y-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Display plot interactively
    save_path : str, optional
        Path to save plot
    figsize : Tuple[float, float], default=(10, 6)
        Figure size (width, height) in inches

    Examples
    --------
    >>> pump_fit, _ = fit_hydraulic_system(current, pump_speed, flow, pressure, back_p)
    >>> plot_pump_fit(current, pump_speed, pump_fit)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot create plot.")
        return

    # Generate fitted curve
    i_range = np.linspace(current.min(), current.max(), 200)
    vp_fit = pump_fit.vpmax * (i_range / pump_fit.imax) ** 2 + pump_fit.vp0

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Raw data
    ax.plot(current, pump_speed, 'b.', markersize=5, alpha=0.5, label='Measured data')
    
    # Fitted curve
    ax.plot(i_range, vp_fit, 'r-', linewidth=2, label='Fitted model')
    
    # Mark Imax if available
    if pump_fit.imax:
        ax.axvline(x=pump_fit.imax, color='g', linestyle='--', alpha=0.7,
                  linewidth=1.5, label=f'Imax = {pump_fit.imax:.0f} A')

    # Mark Vp0 and Vpmax
    ax.axhline(y=pump_fit.vp0, color='orange', linestyle=':', alpha=0.7,
              linewidth=1, label=f'Vp0 = {pump_fit.vp0:.1f} rpm')
    ax.axhline(y=pump_fit.vpmax + pump_fit.vp0, color='purple', linestyle=':', alpha=0.7,
              linewidth=1, label=f'Vpmax = {pump_fit.vpmax + pump_fit.vp0:.1f} rpm')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    else:
        ax.set_title('Pump Speed vs Current', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_flow_pressure_fit(
    current: np.ndarray,
    flow_rate: np.ndarray,
    pressure: np.ndarray,
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
    xlabel: str = "Current (A)",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Plot raw flow and pressure data with fitted curves.

    Parameters
    ----------
    current : np.ndarray
        Current values (A)
    flow_rate : np.ndarray
        Measured flow rate values (l/s)
    pressure : np.ndarray
        Measured pressure values (bar)
    pump_fit : PumpFit
        Fitted pump parameters
    flow_pressure_fit : FlowPressureFit
        Fitted flow and pressure parameters
    xlabel : str, default="Current (A)"
        Label for x-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Display plot interactively
    save_path : str, optional
        Path to save plot
    figsize : Tuple[float, float], default=(12, 5)
        Figure size (width, height) in inches

    Examples
    --------
    >>> pump_fit, flow_pressure_fit = fit_hydraulic_system(current, pump_speed,
    ...                                                     flow, pressure, back_p)
    >>> plot_flow_pressure_fit(current, flow, pressure, pump_fit, flow_pressure_fit)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available. Cannot create plot.")
        return

    # Generate fitted curves
    i_range = np.linspace(current.min(), current.max(), 200)
    vp_fit = pump_fit.vpmax * (i_range / pump_fit.imax) ** 2 + pump_fit.vp0
    
    # Flow: F(I) = F0 + Fmax * Vp(I)/(Vpmax + Vp0)
    flow_fit = flow_pressure_fit.f0 + flow_pressure_fit.fmax * vp_fit / (
        pump_fit.vpmax + pump_fit.vp0
    )
    
    # Pressure: P(I) = Pmin + Pmax * [Vp(I)/(Vpmax + Vp0)]^2
    pressure_fit = flow_pressure_fit.pmin + flow_pressure_fit.pmax * (
        vp_fit / (pump_fit.vpmax + pump_fit.vp0)
    ) ** 2

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Flow Rate
    ax1.plot(current, flow_rate, 'b.', markersize=5, alpha=0.5, label='Measured')
    ax1.plot(i_range, flow_fit, 'r-', linewidth=2, label='Fitted model')
    ax1.axhline(y=flow_pressure_fit.f0, color='orange', linestyle=':', alpha=0.7,
               linewidth=1, label=f'F0 = {flow_pressure_fit.f0:.2f} l/s')
    ax1.axhline(y=flow_pressure_fit.f0 + flow_pressure_fit.fmax, color='purple',
               linestyle=':', alpha=0.7, linewidth=1,
               label=f'Fmax = {flow_pressure_fit.fmax:.2f} l/s')
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel('Flow Rate (l/s)', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Flow Rate vs Current', fontsize=12, fontweight='bold')

    # Plot 2: Pressure
    ax2.plot(current, pressure, 'b.', markersize=5, alpha=0.5, label='Measured')
    ax2.plot(i_range, pressure_fit, 'r-', linewidth=2, label='Fitted model')
    ax2.axhline(y=flow_pressure_fit.pmin, color='orange', linestyle=':', alpha=0.7,
               linewidth=1, label=f'Pmin = {flow_pressure_fit.pmin:.2f} bar')
    ax2.axhline(y=flow_pressure_fit.pmin + flow_pressure_fit.pmax, color='purple',
               linestyle=':', alpha=0.7, linewidth=1,
               label=f'Pmax = {flow_pressure_fit.pmax:.2f} bar')
    if flow_pressure_fit.back_pressure:
        ax2.axhline(y=flow_pressure_fit.back_pressure.mean, color='green',
                   linestyle='--', alpha=0.7, linewidth=1,
                   label=f'Pback = {flow_pressure_fit.back_pressure.mean:.2f} bar')
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel('Pressure (bar)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Pressure vs Current', fontsize=12, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_hysteresis_fit(
    power: np.ndarray,
    flow_rate: np.ndarray,
    hyst_fit: HysteresisFit,
    xlabel: str = "Power (MW)",
    ylabel: str = "Flow Rate (m³/h)",
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Plot raw power-flow data and fitted hysteresis model.

    This is a convenience wrapper around hysteresis.plot_hysteresis_fit()
    that works directly with fitting results.

    Parameters
    ----------
    power : np.ndarray
        Power values (MW)
    flow_rate : np.ndarray
        Flow rate values (m³/h)
    hyst_fit : HysteresisFit
        Fitted hysteresis parameters from fit_hysteresis_parameters()
    xlabel : str, default="Power (MW)"
        Label for x-axis
    ylabel : str, default="Flow Rate (m³/h)"
        Label for y-axis
    title : str, optional
        Plot title
    show : bool, default=True
        Display plot interactively
    save_path : str, optional
        Path to save plot
    figsize : Tuple[float, float], default=(12, 8)
        Figure size (width, height) in inches

    Examples
    --------
    >>> hyst_fit = fit_hysteresis_parameters(power, flow_rate, n_levels=3)
    >>> plot_hysteresis_fit(power, flow_rate, hyst_fit,
    ...                     xlabel='Magnet Power (MW)',
    ...                     ylabel='Cooling Flow (m³/h)')
    """
    try:
        import matplotlib.pyplot as plt
        from .hysteresis import plot_hysteresis_fit as _plot_hyst
    except ImportError:
        logger.warning("matplotlib not available. Cannot create plot.")
        return

    # Create DataFrame for compatibility with hysteresis plotting
    import pandas as pd
    df = pd.DataFrame({"x": power, "y": flow_rate})

    # Use hysteresis module's plotting function
    _plot_hyst(
        df,
        hyst_fit.thresholds,
        hyst_fit.low_values,
        hyst_fit.high_values,
        x_col="x",
        y_col="y",
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        show=show,
        save_path=save_path,
        figsize=figsize,
    )
