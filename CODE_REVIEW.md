# Code Review: python_magnetcooling

*Updated after fixes applied on branch `claude/review-package-0Tahe`.*

---

## What Was Fixed ✅

| Issue | Detail |
|---|---|
| `pextra` dropped in `hcorrelation` | Now `Pextra=pextra` correctly forwarded to `Uw` |
| Montgomery offset `273` → `273.15` | Temperature conversion now matches `correlations.py` |
| Swamee constants | Now `1.325` / `3.7`, matching standard formula |
| `Optional[float]` truthy check in `_compute_channel_uniform` | Uses `is not None` |
| `Optional[float]` truthy check in `_compute_channel_axial` | Uses `is not None` |
| `matplotlib` unconditional import | Deferred to inside `_create_plot` with try/except |
| `y_spec.pop("col")` mutation | Uses `y_spec["col"]` + filtered copy of kwargs |
| `specific_heat_outlet` wrong units | Now `Steam_outlet.cp` [kJ/kg/K]; unit annotations updated |
| `thermohydraulics.py` imported `getDT` from `cooling.py` | Removed; replaced with direct `WaterProperties.compute_temperature_rise()` calls — `thermohydraulics.py` no longer depends on `cooling.py` |
| `compute_from_waterflow()` mutated caller's `ThermalHydraulicInput` | Fixed with `dataclasses.replace()` throughout; original object is never modified |
| Global `simplefilter("ignore")` in `waterflow.py` | Replaced with targeted `warnings.filterwarnings(..., module="pint")` |
| `uguess=0` default in `Uw()` (`cooling.py`) | Changed to `uguess: float = 1.0`; prevents `Re=0 → log(0)` domain error on first iteration |
| Dead bare-string block in `cooling.py` | Removed unimplemented friction variant stubs (karman, rough, gnielinski) |
| Unreachable `importlib_metadata` fallback in `__init__.py` | Removed; package requires Python 3.11+ which ships `importlib.metadata` |
| `fuzzy` not applied in `Dittus`/`Colburn`/`Silverberg` (`cooling.py`) | **Closed by design** — fuzzy is intentionally applied only in `Montgomery` in the legacy interface; `DittusBoelterCorrelation`, `ColburnCorrelation`, `SilverbergCorrelation` in `correlations.py` are the authoritative implementations |

---

## Remaining Bugs

### 1. `ThermalHydraulicCalculator` has no direct unit tests

`test_correlations.py` and `test_friction.py` only cover `MontgomeryCorrelation`,
`ConstantFriction`, and `BlasiusFriction`. The following have no test coverage:

- `DittusBoelterCorrelation`, `ColburnCorrelation`, `SilverbergCorrelation`
- `FilonenkoFriction`, `ColebrookFriction`, `SwameeFriction`
- `ThermalHydraulicCalculator` (no direct unit tests for any cooling level)
- No cross-checking tests between `cooling.py` and `correlations.py`/`friction.py`
- No multi-channel integration test for `outlet_temp_mixed`

---

## Remaining Code Quality Issues

### 3. `heatexchanger_primary.py` — commented-out `heatBalance` function

Lines 587–599 contain a commented-out `heatBalance()` function. Either restore it
or remove it.

### 4. `cooling.py` — ~15 commented-out `print` statements

Scattered debug prints that were commented out but never cleaned up. Remove them.

---

## Quick-fix Checklist

| # | File | Fix |
|---|------|-----|
| 1 | `tests/` | Add tests for `DittusBoelterCorrelation`, `ColburnCorrelation`, `SilverbergCorrelation`, `FilonenkoFriction`, `ColebrookFriction`, `SwameeFriction`, and `ThermalHydraulicCalculator` |
| 2 | `heatexchanger_primary.py` | Remove or restore `heatBalance` dead code |
| 3 | `cooling.py` | Remove ~15 commented-out `print` statements |
