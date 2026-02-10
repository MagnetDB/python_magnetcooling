# Flow Parameter Extraction Pipeline (python_magnetrun)

## Overview

The `flow_params_magnetrun_pipeline.py` script implements the complete pipeline using **python_magnetrun** methods for more advanced fitting:

```
DataFrame (pandas)
      ↓
PWLF Fit: Pump Speed → Automatic Imax detection from breakpoints
      ↓
python_magnetrun.fit: Flow Rate → F = F0 + Fmax·Vp/(Vpmax+Vp0)
      ↓
python_magnetrun.fit: Pressure → P = Pmin + Pmax·[Vp/(Vpmax+Vp0)]²
      ↓
Statistics: Back Pressure
      ↓
Build flow_params Dictionary
      ↓
Use waterflow_factory.from_flow_params()
      ↓
WaterFlow Object → Ready for Calculations
```

## Key Differences from Basic Pipeline

### Basic Pipeline (`flow_params_pipeline.py`)
- Uses `scipy.optimize.curve_fit()` for all fits
- Imax is a fixed parameter
- Simple quadratic fits
- Works with numpy arrays

### MagnetRun Pipeline (`flow_params_magnetrun_pipeline.py`)
- Uses **pwlf** (piecewise linear fitting) for pump speed
- **Automatically detects Imax** from breakpoint detection
- Uses `python_magnetrun.processing.fit` for flow and pressure
- Works with **pandas DataFrame**
- More sophisticated statistical analysis
- Better handles plateau regions in data

## Usage

### Basic Execution

```bash
cd examples
python flow_params_magnetrun_pipeline.py
```

### With Plot Display

```bash
python flow_params_magnetrun_pipeline.py --show-plots
```

### With Debug Output

```bash
python flow_params_magnetrun_pipeline.py --show-plots --debug
```

## What It Does

### Step 1: Generate DataFrame
- Creates pandas DataFrame (magnetrun's preferred format)
- Includes plateau region to test Imax detection
- Columns: current, rpm, flow, pressure, back_pressure

### Step 2: PWLF Pump Speed Fit
**Key Feature: Automatic Imax Detection**

Uses piecewise linear fitting to detect breakpoints:
- Tries 1-2 segments automatically
- If 2 segments found → breakpoint is the detected Imax
- Prints detailed fit statistics table
- Uses `sympy` for symbolic equations

**Model**: 
- Segment 1 (I < Imax): `Vp = a·I² + b·I + c`
- Segment 2 (I > Imax): Plateau (if detected)

**Output**:
```
Pump Speed Fit Results:
  Breakpoints: [0.0, 28032.45, 32000.0]
  Beta coefficients: [...]
  
**DETECTED Imax from breakpoint: 28032 A**

Extracted Parameters:
  Vp0 (at I=0): 1000.23 rpm
  Vpmax (at I=Imax): 2840.15 rpm
  Imax: 28032 A
```

### Steps 3-4: Flow & Pressure Fits
Uses `python_magnetrun.processing.fit()`:
- Same models as basic pipeline
- More robust error handling
- Better integration with magnetrun ecosystem
- Works directly with DataFrame

### Step 5: Back Pressure Statistics
- Filters data up to detected Imax
- Computes mean and standard deviation
- Optional visualization of distribution

### Steps 6-9: Same as Basic Pipeline
- Build flow_params dictionary
- Create WaterFlow using waterflow_factory
- Demonstrate calculations
- Save results

## Dependencies

### Required (from python_magnetrun):
```bash
pip install python-magnetrun
pip install pwlf
pip install sympy
pip install tabulate
```

### Also needs:
- numpy
- pandas
- scipy
- matplotlib
- python_magnetcooling (local package)

## Output Files

1. **flow_params_magnetrun_output.json** - Saved flow parameters
2. **pwlf_pump_speed_fit.png** - PWLF fit visualization (if --show-plots)
3. **back_pressure_stats.png** - Back pressure statistics (if --show-plots)
4. **Flow_fit.png** - Flow rate fit (from python_magnetrun.fit)
5. **Pressure_fit.png** - Pressure fit (from python_magnetrun.fit)

## Automatic Imax Detection Example

The pwlf fitting automatically detects when the pump speed plateaus:

```python
# Data with plateau at Imax
Current:  [0 ... 28000 ... 32000] A
RPM:      [1000 ... 3840 ... 3840] rpm (plateau!)
                        ↑
                   Breakpoint detected → Imax = 28000 A
```

**Fit Results**:
```
Parameter type   Parameter value   Standard error      t        P > |t|
Beta             1000.234          1.567              638.123   0.000
Beta             2.456e-5          1.234e-7           199.123   0.000
Breakpoint       28032.451         45.678             613.234   0.000
```

## Statistical Analysis

The pwlf fit provides comprehensive statistics:
- **Beta coefficients**: Polynomial parameters for each segment
- **Standard errors**: Uncertainty in parameters
- **t-statistics**: Parameter significance
- **p-values**: Statistical significance of parameters
- **Breakpoints**: Automatically detected transition points

## Integration with flow_params_magnetrun.py

This pipeline replicates the logic from `flow_params_magnetrun.py` but:
1. Works with synthetic data (no database connection needed)
2. Clearly separated into steps
3. Uses waterflow_factory for object creation
4. Adds visualization and validation

To use with real data:

```python
from python_magnetrun.magnetdata import MagnetData

# Load real data
data = MagnetData.fromtxt("experimental_data.csv")
df = data.to_dataframe()

# Run pipeline
waterflow = main_with_real_data(df)
```

## Comparison Table

| Feature | Basic Pipeline | MagnetRun Pipeline |
|---------|---------------|-------------------|
| Fitting method | scipy.optimize | pwlf + magnetrun.fit |
| Imax detection | Fixed value | Automatic from data |
| Data format | numpy arrays | pandas DataFrame |
| Pump speed model | Simple quadratic | Piecewise polynomial |
| Plateau handling | No | Yes (automatic) |
| Statistical output | Basic | Comprehensive |
| Integration | Generic | magnetrun ecosystem |

## When to Use Each Pipeline

### Use Basic Pipeline when:
- You have clean data without plateaus
- Imax is known and fixed
- You want simple, fast fitting
- You don't need python_magnetrun features

### Use MagnetRun Pipeline when:
- You need automatic Imax detection
- Your data has plateau regions
- You want detailed statistical analysis
- You're working with magnetrun ecosystem
- You need piecewise fitting capabilities

## Example Output

```
======================================================================
STEP 2: Fit Pump Speed using Piecewise Linear Fitting (pwlf)
======================================================================
Model: Automatic breakpoint detection for Imax

Trying 1 segment(s)...
  Breakpoints: [300.0, 32000.0]
  Beta coefficients: [1000.234, 2.456e-5]
  Final y (predicted): 3840.23, actual: 3840.15
  Error at end: 0.08

Trying 2 segment(s)...
  Breakpoints: [300.0, 28032.45, 32000.0]
  Beta coefficients: [1000.234, 2.456e-5, 3840.123]
  
**DETECTED Imax from breakpoint: 28032 A**

Extracted Parameters:
  Vp0 (at I=0): 1000.23 rpm
  Vpmax (at I=Imax): 2840.15 rpm
  Imax: 28032 A

======================================================================
STEP 3: Fit Flow Rate using python_magnetrun.fit
======================================================================
Model: F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)

Flow Rate Fit Results:
  F0 = -0.1234 l/s
  Fmax = 140.0567 l/s
  Covariance diagonal: [0.0234, 0.0456]
...
```

## Advanced Features

### Custom Breakpoint Guesses

You can provide initial guesses for breakpoints:

```python
# If you suspect Imax is around 28000 A
my_pwlf, eqns, imax = pwlf_fit_pump_speed(
    df,
    current_col='current',
    rpm_col='rpm',
    breakpoint_guess=[28000],
    show=True
)
```

### Multi-segment Fitting

For complex pump behavior with multiple operating regimes:

```python
my_pwlf, eqns, imax = pwlf_fit_pump_speed(
    df,
    current_col='current',
    rpm_col='rpm',
    max_segments=3,  # Try up to 3 segments
    show=True
)
```

## See Also

- `flow_params_pipeline.py` - Basic pipeline using scipy
- `flow_params_magnetrun.py` - Original magnetrun implementation
- `waterflow_factory.py` - Factory functions for WaterFlow creation
- `PIPELINE_README.md` - Documentation for basic pipeline
