# Task: Regression Testing for `python_magnetcooling` — Phase 4 Validation

## Context

We have implemented the `python_magnetcooling` package, which extracts and refactors thermal-hydraulic calculation logic from `python_magnetworkflows`. Before completing Phase 4 (FeelPP integration) and wiring the new package into `python_magnetworkflows/error.py`, we need a rigorous regression testing strategy to guarantee that the new implementation produces identical results to the old one.

This prompt guides a **layered testing approach** — building confidence from isolated function comparisons up to full FeelPP simulation runs.

### Repositories Involved

- **`python_magnetcooling`** — New standalone package (the code under test)
- **`python_magnetworkflows`** — Existing package containing the reference implementation
  - `python_magnetworkflows/cooling.py` — Original correlation/friction/solver functions
  - `python_magnetworkflows/waterflow.py` — Original pump characteristics class
  - `python_magnetworkflows/error.py` — Original FeelPP integration loop

### Naming Conventions in This Prompt

- **"Old"** or **"reference"** = `python_magnetworkflows` implementation
- **"New"** = `python_magnetcooling` implementation
- **"Adapter"** = `python_magnetcooling.feelpp.FeelppThermalHydraulicAdapter`

---

## Layer 1: Unit-Level Regression Tests

**Goal**: Verify that each extracted function/class produces identical results to its original counterpart, in isolation.

**Dependencies**: `python_magnetcooling`, `python_magnetworkflows`, `iapws`, `numpy`, `pint`. No FeelPP required.

**Location**: `python_magnetcooling/tests/test_regression_unit.py`

### 1.1 Water Properties

Compare `python_magnetworkflows/cooling.py` functions against `python_magnetcooling/water_properties.py`:

| Old (`cooling.py`) | New (`water_properties.py`) |
|---|---|
| `steam(Tw, P)` → returns IAPWS97 object | `WaterProperties.get_state(T, P)` → returns `WaterState` |
| `getDT(flow, Power, Tw, P)` | `WaterProperties.compute_temperature_rise(flow, power, T, P)` |
| `Reynolds(Steam, U, Dh, L)` | `WaterProperties.compute_reynolds(velocity, Dh, T, P)` |

**Test matrix** — Use representative operating conditions from actual magnets:

| Condition | Temperature [K] | Pressure [bar] | Notes |
|---|---|---|---|
| Cold inlet | 290.0 | 15.0 | Typical Bitter inlet |
| Warm mid-channel | 305.0 | 12.0 | Mid-channel conditions |
| Hot outlet | 320.0 | 10.0 | High-power outlet |
| Low pressure | 295.0 | 5.0 | Helix conditions |
| High pressure | 300.0 | 22.0 | Maximum pump pressure |

For each condition, compare:
- `Steam.rho` vs `WaterState.density`
- `Steam.cp * 1e3` vs `WaterState.specific_heat` (note: IAPWS returns cp in kJ/kg/K)
- `Steam.k` vs `WaterState.thermal_conductivity`
- `Steam.mu` vs `WaterState.dynamic_viscosity`
- Prandtl number (derived)
- `getDT(...)` vs `compute_temperature_rise(...)` for powers in [10000, 50000, 100000] W
- `Reynolds(...)` vs `compute_reynolds(...)` for velocities in [1, 3, 5, 8, 10] m/s

**Tolerance**: Exact match (< 1e-12 relative error). These wrap the same IAPWS library, so any difference indicates a unit conversion bug.

### 1.2 Heat Transfer Correlations

Compare `python_magnetworkflows/cooling.py` functions against `python_magnetcooling/correlations.py`:

| Old | New |
|---|---|
| `Montgomery(Tw, Pw, dPw, U, Dh, L, friction, fuzzy)` | `MontgomeryCorrelation(fuzzy_factor=fuzzy).compute(T, P, U, Dh, L)` |
| `Dittus(Tw, Pw, dPw, U, Dh, L, friction, fuzzy, pextra)` | `DittusBoelterCorrelation(fuzzy_factor=fuzzy).compute(T, P, U, Dh, L)` |
| `Colburn(Tw, Pw, dPw, U, Dh, L, friction, fuzzy, pextra)` | `ColburnCorrelation(fuzzy_factor=fuzzy).compute(T, P, U, Dh, L)` |
| `Silverberg(Tw, Pw, dPw, U, Dh, L, friction, fuzzy, pextra)` | `SilverbergCorrelation(fuzzy_factor=fuzzy).compute(T, P, U, Dh, L)` |

**Test matrix**:

| Parameter | Values |
|---|---|
| Temperature | 290, 300, 310, 320 K |
| Pressure | 5, 10, 15, 20 bar |
| Velocity | 1, 3, 5, 8, 10 m/s |
| Hydraulic diameter | 0.002, 0.004, 0.008 m |
| Length | 0.2, 0.5, 1.0 m |
| Fuzzy factor | 1.0, 1.2, 1.7 |

**Important notes for Montgomery**:
- The old code uses `Dh` directly in the formula: `exp(log(Dh) * 0.2)` — but the formula expects centimeters
- The new code converts: `dh_cm = hydraulic_diameter * 100.0`
- **Verify which is correct against the original Montgomery reference** (Montgomery, "Solenoid Magnet Design", 1969, p38, eq 3.3)
- If the old code was passing Dh in meters where cm was expected, the new code may intentionally differ — document this

**Important notes for Dittus/Colburn/Silverberg**:
- The old code calls `hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, name)` which internally calls `Uw()` to compute velocity from friction
- The new code takes velocity as a direct input
- Make sure to compare at the **same velocity** — extract U from the old code path and feed it to the new code

**Tolerance**: < 1e-6 relative error for Dittus/Colburn/Silverberg. For Montgomery, document any intentional differences due to unit fix.

### 1.3 Friction Factor Models

Compare `python_magnetworkflows/cooling.py` friction functions against `python_magnetcooling/friction.py`:

| Old | New |
|---|---|
| `Constant(Re, Dh, f, rugosity)` | `ConstantFriction(value=0.055).compute(Re, Dh)` |
| `Blasius(Re, Dh, f, rugosity)` | `BlasiusFriction().compute(Re, Dh)` |
| `Filonenko(Re, Dh, f, rugosity)` | `FilonenkoFriction().compute(Re, Dh)` |
| `Colebrook(Re, Dh, f, rugosity)` | `ColebrookFriction(roughness=rugosity).compute(Re, Dh, f)` |
| `Swamee(Re, Dh, f, rugosity)` | `SwameeFriction(roughness=rugosity).compute(Re, Dh, f)` |

**Note**: The old code spells it `Swanee`. We assume this has been renamed to `Swamee` in `python_magnetworkflows` by the time these tests run.

**Test matrix**:

| Reynolds | Dh [m] | f_guess | Roughness [m] |
|---|---|---|---|
| 5000 | 0.004 | 0.055 | 0.012e-3 |
| 10000 | 0.008 | 0.055 | 0.012e-3 |
| 50000 | 0.004 | 0.03 | 0.012e-3 |
| 100000 | 0.008 | 0.02 | 0.05e-3 |

**Tolerance**: < 1e-6 relative for Constant/Blasius/Filonenko. For Colebrook/Swamee (iterative), < 1e-4 relative (convergence tolerance differences may exist).

**Key check for Colebrook**: The old code uses a different iteration structure (`_iterative_convergence` operating on `1/sqrt(f)`) vs the new code iterating on `f` directly. Verify the converged values match, not the intermediate steps.

### 1.4 Velocity Solver (`Uw`)

Compare `python_magnetworkflows/cooling.py::Uw()` against the velocity computation inside `python_magnetcooling/thermohydraulics.py::_compute_channel_uniform()`.

This is the core iterative solver:
```
U = sqrt(2 * dP / (rho * (Pextra + cf * L/Dh)))
```

**Test approach**: Extract the `Uw()` function's inputs and outputs for a set of conditions, then verify the new code's velocity computation matches.

| Condition | dP [bar] | Dh [m] | L [m] | Friction model | Pextra |
|---|---|---|---|---|---|
| Low flow | 2.0 | 0.004 | 0.3 | Constant | 1.0 |
| Medium flow | 5.0 | 0.008 | 0.5 | Blasius | 1.0 |
| High flow | 10.0 | 0.004 | 0.5 | Colebrook | 1.5 |

Compare: final velocity U, final friction factor cf, number of iterations to converge.

**Tolerance**: < 1e-3 relative for U and cf (matching `error.py`'s own convergence criterion of `1e-3`).

### 1.5 Waterflow / Pump Characteristics

Compare `python_magnetworkflows/waterflow.py::waterflow` against `python_magnetcooling/waterflow.py::WaterFlow`:

| Old method | New method |
|---|---|
| `waterflow.vpump(I)` | `WaterFlow.pump_speed(I)` |
| `waterflow.flow(I)` | `WaterFlow.flow_rate(I)` |
| `waterflow.pressure(I)` | `WaterFlow.pressure(I)` |
| `waterflow.dpressure(I)` | `WaterFlow.pressure_drop(I)` |
| `waterflow.umean(I, S)` | `WaterFlow.velocity(I, S)` |

**Test with identical parameters**:
```python
old = waterflow(Vpump0=1000, Vpmax=2840, F0_l_per_second=0, Fmax_l_per_second=140,
                Pmax=22, Pmin=4, BP=4, Imax=28000)
new = WaterFlow(pump_speed_min=1000, pump_speed_max=2840, flow_min=0, flow_max=140,
                pressure_max=22, pressure_min=4, pressure_back=4, current_max=28000)
```

**Test currents**: 0, 5000, 10000, 14000, 20000, 25000, 28000, 30000 A (including boundary and over-max cases).

**Also test `from_file` / `flow_params`** loading from the same JSON file.

**Tolerance**: Exact match (< 1e-12). Both use the same formulas and pint conversions.

### 1.6 Mixed Outlet Temperature

Compare `python_magnetworkflows/cooling.py::getTout()` against `python_magnetcooling/thermohydraulics.py::_compute_mixed_outlet_temp()`.

**Test cases**: Construct synthetic multi-channel scenarios:

| Case | Channels | T_out [K] | rho [kg/m³] | cp [J/kg/K] | Q [m³/s] |
|---|---|---|---|---|---|
| 2 equal channels | 2 | [300, 310] | [998, 995] | [4180, 4190] | [0.001, 0.001] |
| 3 unequal channels | 3 | [295, 310, 325] | [999, 995, 988] | [4178, 4190, 4200] | [0.002, 0.001, 0.0005] |

**Tolerance**: < 1e-10 relative.

---

## Layer 2: Calculator-Level Regression Tests

**Goal**: Verify that the full `ThermalHydraulicCalculator` produces identical results to the complete iteration loop in `error.py`, for representative magnet configurations, **without FeelPP**.

**Dependencies**: Same as Layer 1. No FeelPP required.

**Location**: `python_magnetcooling/tests/test_regression_calculator.py` and `python_magnetcooling/tests/fixtures/`

### 2.1 Reference Data Generation

We need to capture the inputs and outputs of the old code path for known configurations. There are two approaches:

#### Approach A: Extract from Real Configurations (Preferred)

Use actual magnet configurations (JSON model files) to construct test inputs. For each configuration, manually set up the same parameters that `error.py` would receive:

```python
# Pseudo-code for reference data generation
from python_magnetworkflows.cooling import steam, Uw, getDT, getHeatCoeff, getTout
from python_magnetworkflows.waterflow import waterflow

# Set up like error.py does
flow = waterflow.flow_params("flow_params.json")
Pressure = flow.pressure(abs(objectif))
dPressure = flow.dpressure(abs(objectif))
Umean = flow.umean(abs(objectif), sum(Sh))

# Run the old code path for each channel...
# Save inputs + outputs as JSON fixture
```

#### Approach B: Synthetic Configurations

Construct test inputs that exercise each code path:

**Test Case 1 — Single channel, uniform mode** (simplest path):
```json
{
  "name": "single_channel_uniform",
  "description": "Single Bitter channel, no axial discretization",
  "inputs": {
    "channels": [{
      "hydraulic_diameter": 0.004,
      "cross_section": 2e-5,
      "length": 0.3,
      "power": 30000,
      "temp_inlet": 290.75,
      "temp_outlet_guess": 300.0,
      "heat_coeff_guess": 80000.0
    }],
    "pressure_inlet": 15.0,
    "pressure_drop": 11.0,
    "heat_correlation": "Montgomery",
    "friction_model": "Constant",
    "fuzzy_factor": 1.0,
    "extra_pressure_loss": 1.0
  },
  "expected_outputs": {
    "temp_outlet": null,
    "heat_coeff": null,
    "velocity": null,
    "flow_rate": null,
    "converged": true
  }
}
```

**Test Case 2 — Multiple channels, uniform mode** (Bitter with N channels):
- 10+ channels with different Dh, Sh, L, Power
- Tests the per-channel iteration and mixed outlet temperature

**Test Case 3 — Single channel, axial discretization** (gradHZ mode):
- One channel with FluxZ power distribution (e.g., 20 axial sections)
- Tests `_compute_channel_axial()`

**Test Case 4 — Multiple channels, axial discretization**:
- Combines per-channel + axial discretization
- Most complex code path

**Test Case 5 — With waterflow pump curves**:
- Uses `compute_from_waterflow()` instead of fixed pressures
- Tests the waterflow → pressure → velocity chain

### 2.2 Generating Expected Values

For each test case, run the **old** implementation and record:

```python
# Per channel:
{
  "velocity": U,           # m/s
  "friction_factor": cf,   # dimensionless
  "temp_rise": dTwi,       # K
  "temp_outlet": Ti,       # K
  "heat_coeff": hi,        # W/m²/K
  "flow_rate": Q,          # m³/s
  "iterations": n_iter,    # int
  "converged": True/False
}

# Global:
{
  "total_flow_rate": sum(Q),
  "outlet_temp_mixed": Tout,
  "max_error_temp": err_max_dT,
  "max_error_heat_coeff": err_max_h
}

# For axial discretization, also:
{
  "T_z": [...],    # Temperature distribution along channel
  "h_z": [...]     # Heat coefficient distribution along channel
}
```

### 2.3 Running the Comparison

```python
import json
import pytest
from python_magnetcooling import (
    ThermalHydraulicCalculator, ThermalHydraulicInput,
    ChannelInput, ChannelGeometry, AxialDiscretization
)

@pytest.fixture
def load_reference(request):
    """Load reference data from fixture file"""
    with open(f"tests/fixtures/{request.param}.json") as f:
        return json.load(f)

@pytest.mark.parametrize("load_reference", [
    "single_channel_uniform",
    "multi_channel_bitter",
    "single_channel_axial",
    "multi_channel_axial",
    "waterflow_integration",
], indirect=True)
def test_calculator_matches_reference(load_reference):
    ref = load_reference
    
    # Build inputs from fixture
    calc = ThermalHydraulicCalculator()
    inputs = build_inputs_from_fixture(ref["inputs"])
    
    # Run new calculator
    result = calc.compute(inputs)
    
    # Compare per-channel
    for i, (ch_result, ch_expected) in enumerate(
        zip(result.channels, ref["expected_outputs"]["channels"])
    ):
        assert ch_result.converged == ch_expected["converged"]
        assert abs(ch_result.velocity - ch_expected["velocity"]) / ch_expected["velocity"] < 1e-3
        assert abs(ch_result.temp_outlet - ch_expected["temp_outlet"]) / ch_expected["temp_outlet"] < 1e-4
        assert abs(ch_result.heat_coeff - ch_expected["heat_coeff"]) / ch_expected["heat_coeff"] < 1e-3
    
    # Compare global
    assert abs(result.outlet_temp_mixed - ref["expected_outputs"]["outlet_temp_mixed"]) < 0.01  # K
```

### 2.4 Relaxation Logic Verification

The `error.py` code applies relaxation **after** the inner convergence loop:

```python
dTwi[i] = (1.0 - relax) * tmp_dTwi + relax * dTwH[i]
hi[i] = (1.0 - relax) * tmp_hi + relax * tmp_hi_old
```

Verify that `ThermalHydraulicCalculator` handles the `relaxation_factor` parameter identically. Test with `relax = 0.0` (no relaxation, should match inner loop result exactly) and `relax = 0.3` (typical value).

### 2.5 Edge Cases

- **Zero power channel**: Power = 0 → dT should be 0, h still computed
- **Very high power**: Near boiling conditions → verify no crash
- **Single iteration convergence**: When initial guesses are very close
- **Non-convergence**: What happens when `max_iterations` is reached

---

## Layer 3: FeelPP Integration Tests

**Goal**: Verify that the `FeelppThermalHydraulicAdapter` produces identical results to the current `error.py` when connected to a real FeelPP simulation.

**Dependencies**: `python_magnetcooling`, `python_magnetworkflows`, local FeelPP installation, test magnet configurations with mesh files.

**Location**: `python_magnetcooling/tests/test_regression_feelpp.py` (or a separate integration test directory)

### 3.1 Baseline Capture

**Step 1: Instrument `error.py`**

Add temporary serialization to the **current** `error.py` to capture the complete state at each iteration. This instrumentation should be minimal and removable:

```python
# Add at the top of error.py (temporary)
import json
import os
_CAPTURE_DIR = os.environ.get("MAGNETCOOLING_CAPTURE_DIR", None)
_CAPTURE_IT = 0

def _capture_state(label, data):
    """Serialize state for regression testing"""
    if _CAPTURE_DIR is None:
        return
    global _CAPTURE_IT
    filepath = os.path.join(_CAPTURE_DIR, f"it{_CAPTURE_IT}_{label}.json")
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
```

**Step 2: Capture points**

Insert `_capture_state()` calls at key points in the `compute_error` loop:

```python
# After waterflow setup:
_capture_state("waterflow", {
    "objectif": objectif,
    "Pressure": Pressure,
    "dPressure": dPressure,
    "Umean": Umean,
    "flow": flow.flow(abs(objectif)),
})

# After per-channel computation (inside the channel loop):
_capture_state(f"channel_{i}_{cname}", {
    "Dh": d, "Sh": s, "L": Lh[i],
    "TwH_init": TwH[i], "dTwH_init": dTwH[i], "hwH_init": hwH[i],
    "U_final": U, "cf_final": cf,
    "dTwi_final": dTwi[i], "Ti_final": Ti[i], "hi_final": hi[i],
    "Q_final": Q[i],
    "error_dT": error_dT[-1], "error_h": error_h[-1],
})

# After mixed outlet temp:
_capture_state("mixed_outlet", {
    "List_Tout": List_Tout,
    "List_VolMassout": List_VolMassout,
    "List_SpecHeatout": List_SpecHeatout,
    "List_Qout": List_Qout,
    "Tout_site": Tout_site if len(List_Tout) > 1 else None,
})

# At the end:
_capture_state("final", {
    "err_max": err_max,
    "err_max_dT": err_max_dT,
    "err_max_h": err_max_h,
    "parameters": {k: v for k, v in parameters.items() if isinstance(v, (int, float))},
})
_CAPTURE_IT += 1
```

**Step 3: Run baseline simulation**

```bash
export MAGNETCOOLING_CAPTURE_DIR=/tmp/baseline_capture
mkdir -p $MAGNETCOOLING_CAPTURE_DIR

# Run a short simulation (2-3 iterations) with existing code
# Use a well-known test case, e.g., M9 Bitter or a Helix
python -m python_magnetworkflows.cli --config test_bitter.cfg --itermax 3
```

### 3.2 Test Magnets

Select at least one configuration for each major code path:

| Test Case | Magnet Type | Cooling Mode | Axial Discretization | Channels |
|---|---|---|---|---|
| `test_bitter_uniform` | Bitter | `"H"` in cooling | No FluxZ | ~10-20 slits |
| `test_bitter_gradHZ` | Bitter | `"H"` in cooling | FluxZ | ~10-20 slits |
| `test_helix_uniform` | Helix | `"H"` in cooling | No FluxZ | ~2-4 channels |
| `test_helix_gradHZ` | Helix | `"H"` in cooling | FluxZ | ~2-4 channels |
| `test_global_only` | Any | No `"H"` | No FluxZ | Global only |
| `test_multi_site` | Multi-insert | Both paths | Mixed | Multiple targets |

**Data required per test case**:
- FeelPP configuration file (`.cfg`)
- Mesh files
- Flow parameters JSON
- Any CSV input files referenced by the configuration

### 3.3 Adapter Integration Test

**Step 1: Swap `error.py`**

Create a modified `error.py` that uses the adapter:

```python
# python_magnetworkflows/error_new.py (or modify error.py with a flag)
from python_magnetcooling.feelpp import FeelppThermalHydraulicAdapter
from python_magnetcooling import ThermalHydraulicCalculator

# In compute_error(), replace the thermal-hydraulic section with:
calc = ThermalHydraulicCalculator(verbose=args.debug)
adapter = FeelppThermalHydraulicAdapter(calc)

th_output, params_update, df_update = adapter.compute_from_feelpp_data(
    target, dict_df, p_params, parameters, targets, args, basedir
)
```

**Step 2: Run with adapter**

```bash
export MAGNETCOOLING_CAPTURE_DIR=/tmp/adapter_capture
mkdir -p $MAGNETCOOLING_CAPTURE_DIR

# Run the same simulation with the adapter
python -m python_magnetworkflows.cli --config test_bitter.cfg --itermax 3
```

**Step 3: Compare captures**

```python
# tests/test_regression_feelpp.py
import json
import glob
import pytest

def load_captures(capture_dir):
    """Load all captured states from a directory"""
    captures = {}
    for filepath in sorted(glob.glob(f"{capture_dir}/*.json")):
        name = os.path.basename(filepath).replace(".json", "")
        with open(filepath) as f:
            captures[name] = json.load(f)
    return captures

def test_adapter_matches_baseline():
    baseline = load_captures("/tmp/baseline_capture")
    adapter = load_captures("/tmp/adapter_capture")
    
    # Same number of capture files
    assert set(baseline.keys()) == set(adapter.keys())
    
    for key in baseline:
        b = baseline[key]
        a = adapter[key]
        
        for field in b:
            if isinstance(b[field], (int, float)):
                rel_err = abs(b[field] - a[field]) / max(abs(b[field]), 1e-15)
                assert rel_err < 1e-6, f"{key}.{field}: baseline={b[field]}, adapter={a[field]}, rel_err={rel_err}"
```

### 3.4 Acceptance Criteria

| Quantity | Tolerance | Notes |
|---|---|---|
| Iteration 1 results | < 1e-10 relative | Same initial guesses → should be near-identical |
| Iteration 2+ results | < 1e-6 relative | Small FP differences may accumulate |
| Convergence iteration count | Exact match | Same algorithm should converge in same steps |
| `err_max_dT` | < 1e-6 relative | Critical for FeelPP convergence control |
| `err_max_h` | < 1e-6 relative | Critical for FeelPP convergence control |
| `parameters` dict updates | < 1e-6 relative | These feed back into FeelPP solver |
| `dict_df` DataFrame values | < 1e-4 relative | Rounded to 3 decimal places in old code |
| Mixed outlet temperature | < 0.01 K absolute | Small absolute tolerance |

### 3.5 Investigating Discrepancies

If differences are found:

1. **Check Layer 1 first** — If unit-level tests pass, the issue is in the integration logic
2. **Compare intermediate values** — Add more capture points to narrow down where divergence begins
3. **Check data flow** — The adapter's `_build_input_from_feelpp()` must extract the same values that `error.py` uses
4. **Check parameter naming** — Verify `p_params` key mapping is correct (e.g., `"hwH"` vs `"hw"`, `"dTwH"` vs `"dTw"`)
5. **Check relaxation** — The adapter must apply relaxation at the same point as `error.py`
6. **Check FluxZ handling** — CSV loading, z-position extraction, power distribution indexing
7. **Check dict_df update** — Column naming must match exactly for downstream consumers

---

## Layer 4: Migration Validation (Post-Swap)

**Goal**: After `error.py` is updated to use the adapter, validate that the full workflow still passes all existing tests and produces correct simulation results.

### 4.1 Existing Test Suite

Run all existing `python_magnetworkflows` tests:

```bash
cd python_magnetworkflows
pytest tests/ -v
```

All must pass without modification.

### 4.2 End-to-End Simulation

Run a complete simulation (not just 2-3 iterations) with the updated code:

```bash
# Full convergence run
python -m python_magnetworkflows.cli --config test_bitter.cfg
```

Verify:
- Simulation converges in the same number of global iterations
- Final parameter values match within tolerance
- Output files (CSVs, DataFrames) are identical

### 4.3 Performance Benchmark

Time the thermal-hydraulic computation (not the FeelPP solve) before and after:

```python
import time

# In compute_error, around the thermal-hydraulic section:
t0 = time.perf_counter()
# ... thermal-hydraulic computation ...
t1 = time.perf_counter()
print(f"TH computation: {t1-t0:.4f}s")
```

**Acceptance**: New code should be no more than 20% slower than old code. If significantly slower, profile and optimize.

---

## Implementation Sequence

### Session 1: Layer 1 — Unit Regression Tests

**Deliverables**:
- `tests/test_regression_unit.py` with all comparison tests
- Run results documenting any differences found
- Fix list for any bugs discovered

**Prerequisites**:
- Both packages installed in same environment (`pip install -e .` for both)
- Swanee → Swamee rename done in `python_magnetworkflows`

### Session 2: Layer 2 — Calculator Regression Tests

**Deliverables**:
- `tests/fixtures/` with reference data (JSON files)
- Script to generate reference data from old code
- `tests/test_regression_calculator.py`
- Run results

**Prerequisites**:
- Layer 1 passing (or known differences documented)
- Representative magnet configuration parameters available

### Session 3: Layer 3 — FeelPP Integration Tests

**Deliverables**:
- Instrumentation patch for `error.py`
- Baseline capture data for selected test magnets
- Modified `error.py` using the adapter
- `tests/test_regression_feelpp.py`
- Comparison report

**Prerequisites**:
- Layers 1 and 2 passing
- Local FeelPP installation
- Test magnet configurations with mesh files
- Both packages installed

### Session 4: Layer 4 — Migration and Cleanup

**Deliverables**:
- Final `error.py` using `python_magnetcooling`
- Updated `python_magnetworkflows/pyproject.toml` with `python_magnetcooling` dependency
- Removal of duplicated code from `python_magnetworkflows/cooling.py` (or deprecation)
- Updated documentation
- Migration guide (`docs/migration.rst`)
- Performance benchmark results

**Prerequisites**:
- All layers passing
- Sign-off from stakeholders

---

## Known Issues to Watch For

### 1. Montgomery Unit Convention

The old `Montgomery()` function uses `Dh` directly:
```python
h = fuzzy * 1426.404 * (1 + 1.5e-2 * (Tw - 273)) * exp(log(U) * 0.8) / exp(log(Dh) * 0.2)
```

The new `MontgomeryCorrelation.compute()` converts to centimeters:
```python
dh_cm = hydraulic_diameter * 100.0
h = self.fuzzy_factor * 1426.404 * (1.0 + 0.015 * temp_celsius) * exp(log(velocity) * 0.8) / exp(log(dh_cm) * 0.2)
```

Montgomery's original formula (1969) expects centimeters. If the old code was always passing meters, the new code "fixes" a historical bug. This will produce **different results** — document this intentional change and verify against experimental data.

### 2. `hcorrelation` Internal Velocity Recomputation

In the old `cooling.py`, the `hcorrelation()` helper function internally calls `Uw()` to recompute velocity:
```python
def hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, method):
    Steam = steam(Tw, Pw)
    nU, cf = Uw(Steam, dPw, Dh, L, friction=friction, Pextra=pextra, uguess=U)
    ...
```

This means the old Dittus/Colburn/Silverberg correlations use a **recomputed velocity** (from pressure drop and friction), not the input velocity. The new correlations take velocity as a direct parameter. In the new `thermohydraulics.py` calculator, velocity is computed once per iteration and passed to the correlation — verify this produces the same result.

### 3. Relaxation Timing

In `error.py`, relaxation is applied **after** the inner convergence loop, **outside** the `for it in range(10)` velocity iteration:
```python
dTwi[i] = (1.0 - relax) * tmp_dTwi + relax * dTwH[i]
hi[i] = (1.0 - relax) * tmp_hi + relax * tmp_hi_old
```

Where `dTwH[i]` and `tmp_hi_old` are the **previous outer iteration** values. Verify the adapter preserves this two-level iteration structure.

### 4. FluxZ Column Naming

The old code extracts FluxZ columns with:
```python
key_dz = [fkey for fkey in FluxZ.columns if fkey.endswith(channel_name)]
```

The adapter's `_extract_axial_discretization()` must use the same column naming convention. Verify with actual FluxZ DataFrames.

### 5. `dict_df` Rounding

The old code rounds values when storing in `dict_df`:
```python
dict_df[target]["HeatCoeff"]["hw_" + cname] = [round(hi[i], 3)]
dict_df[target]["DT"]["dTw_" + cname] = [round(dTwi[i], 3)]
dict_df[target]["Uw"]["Uw_" + cname] = [round(U, 3)]
```

The adapter must apply the same rounding. Otherwise, downstream consumers that read from `dict_df` will see slightly different values.

---

## File Inventory

After completing all layers, the test suite should contain:

```
python_magnetcooling/tests/
├── test_regression_unit.py          # Layer 1: function-by-function comparison
├── test_regression_calculator.py    # Layer 2: full calculator comparison
├── test_regression_feelpp.py        # Layer 3: FeelPP integration comparison
├── fixtures/
│   ├── single_channel_uniform.json
│   ├── multi_channel_bitter.json
│   ├── single_channel_axial.json
│   ├── multi_channel_axial.json
│   ├── waterflow_integration.json
│   └── flow_params_test.json
├── conftest.py                      # Shared fixtures and helpers
└── helpers/
    ├── reference_generator.py       # Script to generate Layer 2 fixtures from old code
    └── capture_instrumentation.py   # Patch for error.py baseline capture (Layer 3)
```
