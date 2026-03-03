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
from typing import Optional, List
import logging

import numpy as np

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
