# MagnetRun Pipeline - Summary

Created a new pipeline script `flow_params_magnetrun_pipeline.py` that uses **python_magnetrun** methods for advanced fitting.

## Files Created

1. **examples/flow_params_magnetrun_pipeline.py** (689 lines)
   - Complete pipeline using python_magnetrun methods
   - Automatic Imax detection using pwlf
   - Works with pandas DataFrame
   - 9 clearly defined steps

2. **examples/MAGNETRUN_PIPELINE_README.md**
   - Comprehensive documentation
   - Comparison with basic pipeline
   - Usage examples
   - Advanced features guide

## Key Features

### Automatic Imax Detection
```python
# Uses piecewise linear fitting to detect breakpoint
pwlf_model, eqns, detected_imax = pwlf_fit_pump_speed(df, ...)

# If plateau detected:
**DETECTED Imax from breakpoint: 28032 A**
```

### python_magnetrun Integration
```python
# Step 2: PWLF for pump speed
from python_magnetrun.processing.fit import fit, find_eqn
import pwlf

# Steps 3-4: magnetrun.fit for flow and pressure
params, cov = fit(current_col, flow_col, "Flow", imax, flow_func, df, ...)
```

### Comprehensive Statistics
- Beta coefficients with standard errors
- t-statistics for parameter significance
- p-values for statistical tests
- Breakpoint detection quality metrics

## Usage

```bash
cd examples

# Basic run
python flow_params_magnetrun_pipeline.py

# With plots
python flow_params_magnetrun_pipeline.py --show-plots

# With debug output
python flow_params_magnetrun_pipeline.py --show-plots --debug
```

## Pipeline Steps

```
1. Generate DataFrame (pandas) with plateau region
2. PWLF fit → Automatic Imax detection from breakpoints
3. magnetrun.fit → Flow rate (F0, Fmax)
4. magnetrun.fit → Pressure (Pmin, Pmax)
5. Statistics → Back pressure (mean, std)
6. Build flow_params dictionary
7. waterflow_factory.from_flow_params() → WaterFlow object
8. Demonstrate hydraulic calculations
9. Save results to JSON
```

## Comparison with Basic Pipeline

| Feature | Basic | MagnetRun |
|---------|-------|-----------|
| **Fitting** | scipy.optimize | pwlf + magnetrun |
| **Imax** | Fixed | Auto-detected |
| **Data** | numpy arrays | pandas DataFrame |
| **Pump speed** | Simple quadratic | Piecewise polynomial |
| **Plateaus** | No | Yes (automatic) |
| **Stats** | Basic | Comprehensive |

## Dependencies

Additional packages needed:
```bash
pip install python-magnetrun  # Main fitting framework
pip install pwlf              # Piecewise linear fitting
pip install sympy             # Symbolic equations
pip install tabulate          # Pretty tables
```

## Output Example

```
======================================================================
STEP 2: Fit Pump Speed using Piecewise Linear Fitting (pwlf)
======================================================================

Trying 2 segment(s)...
  Breakpoints: [300.0, 28032.45, 32000.0]
  
Parameter type   Parameter value   Standard error      t        P > |t|
Beta             1000.234          1.567              638.123   0.000
Beta             2.456e-5          1.234e-7           199.123   0.000
Breakpoint       28032.451         45.678             613.234   0.000

**DETECTED Imax from breakpoint: 28032 A**

Extracted Parameters:
  Vp0 (at I=0): 1000.23 rpm
  Vpmax (at I=Imax): 2840.15 rpm
  Imax: 28032 A
```

## Integration Points

Both pipelines use the **same waterflow_factory**:
```python
# After fitting (either pipeline):
waterflow = from_flow_params(flow_params)

# Rest is identical:
velocity = waterflow.velocity(current, cross_section)
```

This maintains clean separation:
- **Data acquisition + fitting**: pipeline scripts
- **Object creation**: waterflow_factory 
- **Calculations**: WaterFlow class

## When to Use

### Use MagnetRun Pipeline when:
✓ Need automatic Imax detection  
✓ Have plateau regions in data  
✓ Want detailed statistical analysis  
✓ Working with magnetrun ecosystem  
✓ Need piecewise fitting

### Use Basic Pipeline when:
✓ Imax is known and fixed  
✓ Simple quadratic fits sufficient  
✓ No python_magnetrun dependency  
✓ Clean data without plateaus

Both pipelines produce the same WaterFlow objects via waterflow_factory!
