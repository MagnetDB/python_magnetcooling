# Prompt: Add Fitting Module to `python_magnetcooling`

## Objective

Implement a `fitting` module in `python_magnetcooling` that provides pure-numerical fitting functions for pump speed, flow rate, and pressure curves. These functions accept NumPy arrays (not MagnetRun objects, not TDMS files) and produce typed result dataclasses that feed into `WaterFlow` construction.

This module is the missing link between raw experimental measurements and the existing `WaterFlow` / `waterflow_factory` infrastructure.

---

## Context and Motivation

Currently, all fitting logic lives in `python_magnetrun` (in `examples/flow_params.py`, `examples/flow_params_pipeline.py`, `examples/flow_params_magnetrun_pipeline.py`, and `examples/flow_params_magnetrun.py`). This creates an undesirable coupling: anyone wanting to characterize a pump system must depend on `python_magnetrun` even if their data doesn't come from LNCMI's acquisition system.

The goal is:

- `python_magnetcooling.fitting` takes **arrays** and returns **typed fit results + WaterFlow objects**
- `python_magnetrun` handles **data extraction** from MagnetRun files and calls into `python_magnetcooling.fitting`
- Users with data from other sources (CSV, manual measurements, simulation) can use `python_magnetcooling.fitting` directly

---

## Physical Models to Implement

All models below are currently implemented as inline closures in the `python_magnetrun` fitting scripts. Extract them as standalone, tested functions.

### 1. Pump Speed vs Current

```
Vp(I) = Vpmax · (I / Imax)² + Vp0
```

Two fitting approaches must be supported:

**Simple quadratic fit** (scipy): fits `Vpmax` and `Vp0` given a known `Imax`.

**Piecewise linear fit** (pwlf): fits 1 or 2 segments with automatic breakpoint detection. When 2 segments are found, the breakpoint gives the detected `Imax`. This is the more advanced method and requires `pwlf` as a dependency.

### 2. Flow Rate vs Current

```
F(I) = F0 + Fmax · Vp(I) / (Vpmax + Vp0)
```

where `Vp(I)` uses the pump speed model above. Fits `F0` and `Fmax` using `scipy.optimize.curve_fit`.

### 3. Pressure vs Current

```
P(I) = Pmin + Pmax · [Vp(I) / (Vpmax + Vp0)]²
```

Fits `Pmin` and `Pmax` using `scipy.optimize.curve_fit`.

### 4. Back Pressure

Not fitted — computed as statistics (mean, std) from the back pressure array.

---

## Module Structure

Create the following file:

```
python_magnetcooling/
├── fitting.py          # NEW — all fitting logic
├── __init__.py         # UPDATE — add new exports
├── waterflow.py        # EXISTING — unchanged
├── waterflow_factory.py # EXISTING — update to use fitting types
└── ...
```

A single `fitting.py` file is sufficient. The module is not large enough to warrant a subpackage.

---

## Detailed Implementation Specification

### Result Dataclasses

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass(frozen=True)
class FitResult:
    """Base result for any curve fit.

    Attributes
    ----------
    parameters : np.ndarray
        Fitted parameter values.
    standard_errors : np.ndarray
        Standard errors on each parameter.
    r_squared : float
        Coefficient of determination.
    residuals : np.ndarray
        Residual array (y_data - y_fit).
    """
    parameters: np.ndarray
    standard_errors: np.ndarray
    r_squared: float
    residuals: np.ndarray


@dataclass(frozen=True)
class PumpSpeedFit:
    """Result of pump speed curve fitting.

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
    fit_result : FitResult
        Underlying fit statistics.
    breakpoints : list of float, optional
        Piecewise breakpoints (only for piecewise method).
    equations : list, optional
        Symbolic piecewise equations (only for piecewise method).
    """
    vpmax: float
    vp0: float
    imax: float
    imax_detected: bool = False
    fit_result: Optional[FitResult] = None
    breakpoints: Optional[List[float]] = None
    equations: Optional[list] = None

    def pump_speed(self, current: float) -> float:
        """Evaluate Vp(I) = Vpmax·(I/Imax)² + Vp0."""
        if current >= self.imax:
            return self.vpmax + self.vp0
        return self.vpmax * (current / self.imax) ** 2 + self.vp0


@dataclass(frozen=True)
class FlowPressureFit:
    """Result of flow rate and pressure curve fitting.

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
        Std deviation of back pressure [bar].
    flow_fit : FitResult
        Fit statistics for the flow curve.
    pressure_fit : FitResult
        Fit statistics for the pressure curve.
    """
    f0: float
    fmax: float
    pmin: float
    pmax: float
    back_pressure: float
    back_pressure_std: float = 0.0
    flow_fit: Optional[FitResult] = None
    pressure_fit: Optional[FitResult] = None
```

### Fitting Functions

```python
def fit_pump_speed_simple(
    current: np.ndarray,
    pump_speed: np.ndarray,
    imax: float,
) -> PumpSpeedFit:
    """
    Fit pump speed vs current using simple quadratic model.

    Model: Vp(I) = Vpmax·(I/Imax)² + Vp0

    Uses scipy.optimize.curve_fit. Requires imax to be known.

    Parameters
    ----------
    current : np.ndarray
        Current values [A]. Should be filtered (e.g., I >= 300 A).
    pump_speed : np.ndarray
        Pump speed values [rpm].
    imax : float
        Maximum operating current [A].

    Returns
    -------
    PumpSpeedFit
    """


def fit_pump_speed_piecewise(
    current: np.ndarray,
    pump_speed: np.ndarray,
    max_segments: int = 2,
    degree: int = 2,
) -> PumpSpeedFit:
    """
    Fit pump speed vs current using piecewise linear fitting (pwlf).

    Automatically detects Imax from breakpoints when 2 segments are used.
    If the second segment is approximately flat (plateau), the breakpoint
    between segments is reported as the detected Imax.

    Parameters
    ----------
    current : np.ndarray
        Current values [A].
    pump_speed : np.ndarray
        Pump speed values [rpm].
    max_segments : int
        Maximum number of segments to try (1 or 2).
    degree : int
        Polynomial degree within each segment (default 2).

    Returns
    -------
    PumpSpeedFit
        With imax_detected=True if breakpoint detection succeeded.

    Raises
    ------
    ImportError
        If pwlf is not installed.
    """


def fit_flow_rate(
    current: np.ndarray,
    flow_rate: np.ndarray,
    pump_fit: PumpSpeedFit,
) -> FitResult:
    """
    Fit flow rate vs current.

    Model: F(I) = F0 + Fmax · Vp(I) / (Vpmax + Vp0)

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
        With parameters = [F0, Fmax].
    """


def fit_pressure(
    current: np.ndarray,
    pressure: np.ndarray,
    pump_fit: PumpSpeedFit,
) -> FitResult:
    """
    Fit inlet pressure vs current.

    Model: P(I) = Pmin + Pmax · [Vp(I) / (Vpmax + Vp0)]²

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
        With parameters = [Pmin, Pmax].
    """


def compute_back_pressure_stats(
    back_pressure: np.ndarray,
) -> tuple[float, float]:
    """
    Compute mean and standard deviation of back pressure.

    Parameters
    ----------
    back_pressure : np.ndarray
        Back pressure measurements [bar].

    Returns
    -------
    (mean, std)
    """
```

### Orchestration Function

```python
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
    experimental arrays.

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
    method : str
        "simple" for scipy quadratic, "piecewise" for pwlf with breakpoint detection.
    current_threshold : float
        Minimum current for filtering data (default 300 A).

    Returns
    -------
    (PumpSpeedFit, FlowPressureFit)
    """
```

### WaterFlow Construction

```python
def build_waterflow(
    pump_fit: PumpSpeedFit,
    flow_pressure_fit: FlowPressureFit,
) -> "WaterFlow":
    """
    Construct a WaterFlow object from fitted parameters.

    This replaces manual dict construction and waterflow_factory.from_flow_params()
    for the typed path.

    Parameters
    ----------
    pump_fit : PumpSpeedFit
    flow_pressure_fit : FlowPressureFit

    Returns
    -------
    WaterFlow
    """
    from .waterflow import WaterFlow

    return WaterFlow(
        pump_speed_min=pump_fit.vp0,
        pump_speed_max=pump_fit.vpmax,
        flow_min=flow_pressure_fit.f0,
        flow_max=flow_pressure_fit.fmax,
        pressure_max=flow_pressure_fit.pmax,
        pressure_min=flow_pressure_fit.pmin,
        pressure_back=flow_pressure_fit.back_pressure,
        current_max=pump_fit.imax,
    )
```

---

## Updates to Existing Files

### `__init__.py`

Add the following exports:

```python
from .fitting import (
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
```

Add all names to `__all__`.

### `waterflow_factory.py`

Add a new factory function that accepts the typed fit objects:

```python
def from_fits(
    pump_fit: "PumpSpeedFit",
    flow_pressure_fit: "FlowPressureFit",
) -> WaterFlow:
    """
    Create WaterFlow from fitting result dataclasses.

    This is the preferred path for new code. The existing
    from_flow_params() remains for backward compatibility with
    the legacy dict format.
    """
    from .fitting import build_waterflow
    return build_waterflow(pump_fit, flow_pressure_fit)
```

The existing `from_flow_params()`, `from_database_record()`, and `from_fitted_data()` remain unchanged for backward compatibility.

---

## Dependencies

### Required (already in pyproject.toml)

- `numpy` — array operations
- `scipy` — `scipy.optimize.curve_fit` for simple fits

### Optional (new)

Add `pwlf` as an optional dependency:

```toml
[project.optional-dependencies]
fitting = [
    "pwlf>=2.4.0",
    "sympy>=1.11.1",
]
```

`fit_pump_speed_piecewise` should raise `ImportError` with a clear message if `pwlf` is not installed. `sympy` is used for symbolic equation extraction from pwlf fits (the `find_eqn` helper).

---

## Tests

Create `tests/test_fitting.py` with the following test structure:

### Synthetic Test Data

Generate synthetic data from known parameters so expected outputs are verifiable:

```python
import numpy as np

# Known parameters
KNOWN_VPMAX = 2840.0  # rpm
KNOWN_VP0 = 1000.0    # rpm
KNOWN_IMAX = 28000.0  # A
KNOWN_F0 = 0.0        # l/s
KNOWN_FMAX = 140.0    # l/s
KNOWN_PMIN = 4.0       # bar
KNOWN_PMAX = 22.0      # bar
KNOWN_BP = 4.0         # bar

def make_synthetic_data(n=200, noise_level=0.01, seed=42):
    """Generate synthetic hydraulic data with optional noise."""
    rng = np.random.default_rng(seed)
    current = np.linspace(300, 32000, n)

    # True models
    vp = KNOWN_VPMAX * (current / KNOWN_IMAX) ** 2 + KNOWN_VP0
    vp_ratio = vp / (KNOWN_VPMAX + KNOWN_VP0)
    flow = KNOWN_F0 + KNOWN_FMAX * vp_ratio
    pressure = KNOWN_PMIN + KNOWN_PMAX * vp_ratio ** 2
    back_pressure = np.full(n, KNOWN_BP)

    # Add noise
    vp += rng.normal(0, noise_level * KNOWN_VPMAX, n)
    flow += rng.normal(0, noise_level * KNOWN_FMAX, n)
    pressure += rng.normal(0, noise_level * KNOWN_PMAX, n)
    back_pressure += rng.normal(0, 0.1, n)

    return current, vp, flow, pressure, back_pressure
```

### Test Cases

1. **`test_fit_pump_speed_simple_noiseless`** — Verify exact recovery of Vpmax, Vp0 from clean data (tolerance < 1e-6).
2. **`test_fit_pump_speed_simple_noisy`** — Verify recovery within 2% from noisy data.
3. **`test_fit_pump_speed_piecewise`** — With plateau data appended beyond Imax, verify Imax is detected within 5% of true value and `imax_detected is True`.
4. **`test_fit_pump_speed_piecewise_import_error`** — Mock `pwlf` unavailable, verify `ImportError` with helpful message.
5. **`test_fit_flow_rate`** — Given a correct PumpSpeedFit, verify F0 and Fmax recovery.
6. **`test_fit_pressure`** — Given a correct PumpSpeedFit, verify Pmin and Pmax recovery.
7. **`test_compute_back_pressure_stats`** — Verify mean and std against `np.mean`/`np.std`.
8. **`test_fit_hydraulic_system_simple`** — End-to-end with method="simple".
9. **`test_fit_hydraulic_system_piecewise`** — End-to-end with method="piecewise".
10. **`test_build_waterflow`** — Verify the returned WaterFlow has correct attributes and its methods (`.pump_speed()`, `.flow_rate()`, `.pressure()`) produce expected values.
11. **`test_waterflow_factory_from_fits`** — Verify `waterflow_factory.from_fits()` produces identical WaterFlow to `build_waterflow()`.
12. **`test_fit_result_r_squared`** — Verify R² ≈ 1.0 for noiseless data, R² < 1.0 for noisy data.
13. **`test_current_threshold_filtering`** — Verify that `fit_hydraulic_system` filters data below threshold.

---

## Implementation Notes

### `find_eqn` Helper

The existing code in `python_magnetrun` uses a `find_eqn(my_pwlf)` function that extracts symbolic equations from a pwlf model using sympy. This should be copied into `fitting.py` as a private helper `_extract_piecewise_equations(pwlf_model)`. It is only used when `method="piecewise"`.

### Logging, Not Printing

Replace all `print()` statements from the original pipeline code with `logging.getLogger("magnetcooling.fitting")`. The fitting functions should be silent by default and log at DEBUG level.

### No Plotting

The fitting module must not contain any matplotlib code. Plotting is the caller's responsibility. The `FitResult` dataclass provides everything needed (parameters, residuals, R²) for the caller to plot if desired.

### Validation

- `current` and dependent arrays must have the same length.
- `current` must contain at least 3 points after threshold filtering.
- `imax` must be positive (for simple method) or auto-detected (for piecewise).
- Raise `ValueError` with descriptive messages for invalid inputs.

---

## Verification Checklist

Before considering this task complete:

- [ ] `fitting.py` created with all dataclasses and functions listed above
- [ ] `__init__.py` updated with new exports in `__all__`
- [ ] `waterflow_factory.py` updated with `from_fits()` function
- [ ] `pyproject.toml` updated with `[fitting]` optional dependency group
- [ ] `tests/test_fitting.py` passes with all 13+ test cases
- [ ] No `print()` statements — logging only
- [ ] No matplotlib imports
- [ ] No imports from `python_magnetrun`
- [ ] `pwlf` import is guarded (lazy, with `ImportError` fallback)
- [ ] All public functions have NumPy-style docstrings with Parameters/Returns/Raises
- [ ] All public functions have type hints
- [ ] `mypy` passes (or near-passes) on `fitting.py`
- [ ] Existing tests in `tests/test_waterflow_factory.py` still pass
