# Code Review: python_magnetcooling

## Summary

Overall the codebase is reasonably well-structured with a clear domain model and good
separation of concerns. The new OOP-based modules (`correlations.py`, `friction.py`,
`water_properties.py`) are clean and well-documented. The legacy `cooling.py` module,
however, contains several bugs and design issues that are partially masked by the
parallel implementations.

The most critical issues are listed first.

---

## Critical Bugs

### 1. `pextra` parameter silently ignored in `hcorrelation` (`cooling.py:314`)

`hcorrelation` accepts a `pextra` parameter but always passes `Pextra=1` to `Uw`,
discarding the caller's value:

```python
# cooling.py:314
nU, _ = Uw(
    Steam,
    dPw,
    Dh,
    L,
    friction,
    Pextra=1,       # Bug: hardcoded, ignores `pextra` argument
    fguess=0.055,
    uguess=U,
    rugosity=rugosity,
)
```

`getHeatCoeff` in the same file correctly passes `pextra` to the correlation
functions, but those functions pass it to `hcorrelation`, where it is silently
dropped. The `extra_pressure_loss` field on `ThermalHydraulicInput` therefore has
**no effect** for non-Montgomery correlations.

### 2. Truthy check on `Optional[float]` fields (`thermohydraulics.py:212,216,275,371`)

Several places use `if channel.velocity_guess:` or `if channel.temp_outlet_guess:`
to detect "not set". For `Optional[float]` fields, the correct check is `is not None`.
A user-supplied value of `0.0` would evaluate as falsy and silently fall back to the
default, leading to wrong results with no error:

```python
# thermohydraulics.py:275 — same pattern at 212, 216, 371
U = channel.velocity_guess if channel.velocity_guess else 5.0      # BUG
U = channel.velocity_guess if channel.velocity_guess is not None else 5.0  # correct
```

### 3. Montgomery temperature conversion inconsistency (`cooling.py:48` vs `correlations.py:99`)

`cooling.py` uses `Tw - 273` (integer offset) while `correlations.py` correctly uses
`temperature - 273.15`. This gives a 0.15 K discrepancy in the temperature term used
to compute the heat transfer coefficient. The two implementations of the same formula
will produce different results:

```python
# cooling.py:48
h = fuzzy * 1426.404 * (1 + 1.5e-2 * (Tw - 273)) * ...

# correlations.py:99 — correct
temp_celsius = temperature - 273.15
```

### 4. `fuzzy` factor unused in Dittus/Colburn/Silverberg (`cooling.py:53-98`)

These functions accept a `fuzzy` parameter but never pass it to `hcorrelation`,
so the correction factor has no effect on any non-Montgomery correlation:

```python
def Dittus(Tw, Pw, dPw, U, Dh, L, friction, fuzzy: float = 1.0, pextra: float = 1):
    params = (0.023, 0.8, 0.4)
    h = hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, "Dittus")
    # fuzzy is silently ignored
    return h
```

---

## Bugs

### 5. Swamee-Jain constants differ between implementations

`cooling.py:Swamee` and `friction.py:SwameeFriction` use different constants:

| Location | Coefficient | Roughness factor |
|---|---|---|
| `cooling.py:196` | `1.3254` | `3.75` |
| `friction.py:189` | `1.325` | `3.7` |
| Standard formula | `1.325` | `3.7` |

`friction.py` matches the standard formula; `cooling.py` deviates.

Additionally, `cooling.py:Swamee` wraps an explicit formula inside
`_iterative_convergence` with a `compute_new` that ignores its argument — the
iteration always produces the same value and converges trivially after 1 step.
This is misleading and wasteful.

### 6. `matplotlib` unconditional import in `heatexchanger_primary.py` (`line 17`)

`matplotlib` is listed as an **optional** dependency (under the `viz` extra), but it
is imported unconditionally at the top of the module:

```python
import matplotlib.pyplot as plt
```

Any code that does `from python_magnetcooling.heatexchanger_primary import heatexchange`
on a system without `matplotlib` will get an `ImportError`, even if the caller never
uses any plotting function. The import should be guarded or deferred to the functions
that need it.

### 7. Mutation of caller-supplied dict in `_create_plot` (`heatexchanger_primary.py:58`)

`y_spec.pop("col")` modifies the dictionaries in the `y_cols` list in-place, mutating
the caller's data. Any code that reuses the same list would get a `KeyError` on the
second call:

```python
for y_spec in y_cols:
    col = y_spec.pop("col")   # mutates caller's dict
    df.plot(x=x_col, y=col, ax=ax, **y_spec)
```

Replace with `y_spec.get("col")` / create a copy: `col = y_spec["col"]` and pass
`{k: v for k, v in y_spec.items() if k != "col"}` to `df.plot`.

### 8. `_compute_mixed_outlet_temp` has an unused parameter (`thermohydraulics.py:462`)

```python
def _compute_mixed_outlet_temp(self, channels: List[ChannelOutput], pressure: float) -> float:
```

`pressure` is accepted but never used inside the method. Remove it or use it.

### 9. `uguess=0` default in `Uw` can cause domain errors (`cooling.py:212`)

Some friction models call `log(Re)` or `log10(Re)`. With `uguess=0`, `U=0`,
`Re = rho*0*Dh/mu = 0`, and `log(0)` raises `ValueError`. The default should be a
small positive value, or the function should validate the input before the loop.

---

## Design Issues

### 10. Duplicate class definitions across two modules

`ChannelGeometry`, `ChannelInput`, `ChannelOutput`, and `AxialDiscretization` are
defined in **both** `thermohydraulics.py` and `channel.py`. `__init__.py` exports
from `channel.py`. `thermohydraulics.py` uses its own local definitions (which lack
validation logic present in `channel.py`). This creates two parallel, incompatible
class hierarchies with the same names. The `thermohydraulics.py` definitions should
be removed and replaced with imports from `channel.py`.

### 11. New OOP modules (`correlations.py`, `friction.py`) not wired into the main solver

`ThermalHydraulicCalculator` delegates entirely to the legacy function-based API in
`cooling.py` via `getHeatCoeff` and `Uw`. The clean OOP implementations in
`correlations.py` and `friction.py` are effectively unused in the main computation
path. Either the new modules should replace the legacy ones, or the legacy ones should
delegate to the new ones — the current dual-implementation is a maintenance burden.

### 12. `compute_from_waterflow` mutates its input (`thermohydraulics.py:251-252`)

```python
inputs.pressure_inlet = waterflow_params.pressure(current)
inputs.pressure_drop = waterflow_params.pressure_drop(current)
```

Directly modifying the caller's `ThermalHydraulicInput` is surprising. Use a copy
(e.g., `dataclasses.replace(inputs, pressure_inlet=..., pressure_drop=...)`) or
clearly document the mutation and rename to `compute_from_waterflow_inplace`.

### 13. Global warning suppression in `waterflow.py` (`lines 18-19`)

```python
simplefilter("ignore")
Quantity([])
```

`simplefilter("ignore")` suppresses **all** warnings globally, not just pint-related
ones. This runs on every import of `waterflow.py` and can hide important warnings
from other libraries. Use a context manager or target only the specific pint warning
with `warnings.filterwarnings`.

---

## Code Quality Issues

### 14. Obscure power-as-exponential idiom throughout `cooling.py`

All power computations use `exp(log(x) * n)` instead of the simpler, faster, and
more readable `x**n`:

```python
# cooling.py:48, 94, 133, 196, 259 — and friction.py:94, 116, 189
exp(log(U) * 0.8)       # should be U**0.8
exp(log(Re) * 0.25)     # should be Re**0.25
```

While mathematically equivalent, the `exp`/`log` form is slower, harder to read,
and will raise `ValueError` for non-positive inputs that `**` would handle correctly
(e.g., `(-1)**2` works but `exp(log(-1)*2)` raises).

### 15. Commented-out code left in production files

- `cooling.py:47` — `# fuzzy = 1.7`
- `cooling.py:107-129` — large block of unimplemented friction methods left as a bare
  string expression (not even a docstring)
- `cooling.py` — ~15 commented-out `print` statements
- `heatexchanger_primary.py:587-599` — commented-out `heatBalance` function

These should be removed or tracked in an issue.

### 16. Dead import fallback in `__init__.py` (`lines 38-39`)

The `importlib_metadata` fallback is dead code — `importlib.metadata` has been
in the standard library since Python 3.8 and the project requires Python ≥ 3.11:

```python
except ImportError:
    # Fallback for Python < 3.8 (though we require 3.11+)
    from importlib_metadata import version, PackageNotFoundError
```

### 17. Unit conversion via pint on every `flow_rate()` call (`waterflow.py:137-143`)

```python
units = [ureg.liter / ureg.second, ureg.meter**3 / ureg.second]
F0 = Quantity(self.flow_min, units[0]).to(units[1]).magnitude
Fmax = Quantity(self.flow_max, units[0]).to(units[1]).magnitude
```

This creates `UnitRegistry` unit objects and `Quantity` instances on every call.
A litre is exactly `1e-3 m³`; a simple multiply suffices:

```python
F0 = self.flow_min * 1e-3
Fmax = self.flow_max * 1e-3
```

### 18. Mixed language identifiers (French/English)

Several functions and DataFrame column names use French terminology:
`debitbrut`, `debitc`, `debith`, `tsb`, `teb`. While this reflects the domain
language used at LNCMI, it reduces readability for international contributors.
At minimum, docstrings should translate all French terms to English.

### 19. Missing `__all__` in most modules

Only `__init__.py` defines `__all__`. Modules like `cooling.py`, `correlations.py`,
`friction.py` expose all their internals. Explicitly defining `__all__` in each
module would clarify the public API and prevent accidental `from module import *`
pollution.

### 20. Incomplete test coverage

- `test_correlations.py` only tests `MontgomeryCorrelation`; `DittusBoelterCorrelation`,
  `ColburnCorrelation`, and `SilverbergCorrelation` have no tests.
- `test_friction.py` only tests `ConstantFriction` and `BlasiusFriction`;
  `FilonenkoFriction`, `ColebrookFriction`, and `SwameeFriction` have no tests.
- `test_channel.py` is described as a placeholder.
- `ThermalHydraulicCalculator` has no direct unit tests.
- The two-implementation divergence (bug #3, #5) is not caught because there are no
  tests comparing `cooling.py` output to `correlations.py`/`friction.py` output.

---

## Quick-fix Checklist

| # | File | Line(s) | Fix |
|---|------|---------|-----|
| 1 | `cooling.py` | 314 | Pass `pextra` to `Uw` instead of hardcoding `1` |
| 2 | `thermohydraulics.py` | 212,216,275,371 | Replace `if x:` with `if x is not None:` |
| 3 | `cooling.py` | 48 | Use `Tw - 273.15` |
| 4 | `cooling.py` | 65,81,97 | Pass `fuzzy` to `hcorrelation` |
| 5 | `cooling.py` | 196 | Fix constants to match standard formula (`1.325`, `3.7`) |
| 6 | `heatexchanger_primary.py` | 17 | Guard `matplotlib` import |
| 7 | `heatexchanger_primary.py` | 58 | Use `col = y_spec["col"]`, pass filtered dict |
| 8 | `thermohydraulics.py` | 462 | Remove unused `pressure` parameter |
| 9 | `cooling.py` | 212 | Change `uguess` default or add `U > 0` guard |
| 13 | `waterflow.py` | 18-19 | Replace global `simplefilter` with targeted filter |
| 16 | `__init__.py` | 38-39 | Remove dead `importlib_metadata` fallback |
| 17 | `waterflow.py` | 137-143 | Replace pint conversion with `* 1e-3` |
