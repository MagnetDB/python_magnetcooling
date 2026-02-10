# Flow Parameter Extraction Pipeline

## Overview

The `flow_params_pipeline.py` script implements the complete visual pipeline for extracting flow parameters and creating WaterFlow objects:

```
Database Records
      ↓
Load Experimental Data (Rpm, Flow, Pressure vs Current)
      ↓
Fit #1: Pump Speed    → Vp = Vpmax·(I/Imax)² + Vp0
      ↓
Fit #2: Flow Rate     → F = F0 + Fmax·Vp/(Vpmax+Vp0)
      ↓
Fit #3: Pressure      → P = Pmin + Pmax·[Vp/(Vpmax+Vp0)]²
      ↓
Build flow_params Dictionary
      ↓
Use waterflow_factory.from_flow_params()
      ↓
WaterFlow Object → Ready for Calculations
```

## Usage

### Basic Execution

```bash
cd examples
python flow_params_pipeline.py
```

### With Plot Display

```bash
python flow_params_pipeline.py --show-plots
```

## What It Does

### Step 1: Load Experimental Data
- In real workflow: loads from database records
- In this script: generates synthetic data with realistic noise

### Steps 2-4: Curve Fitting
- **Pump Speed Fit**: Uses `scipy.optimize.curve_fit()` to find Vpmax and Vp0
- **Flow Rate Fit**: Fits F0 and Fmax using the pump speed relationship
- **Pressure Fit**: Fits Pmin and Pmax as quadratic function of pump speed

### Step 5: Back Pressure Statistics
- Calculates mean and standard deviation of back pressure

### Step 6: Build flow_params Dictionary
- Assembles all fitted parameters into the standard format:
```json
{
  "Vp0": {"value": 1000, "unit": "rpm"},
  "Vpmax": {"value": 2840, "unit": "rpm"},
  "F0": {"value": 0, "unit": "l/s"},
  "Fmax": {"value": 140, "unit": "l/s"},
  "Pmax": {"value": 22, "unit": "bar"},
  "Pmin": {"value": 4, "unit": "bar"},
  "BP": {"value": 4, "unit": "bar"},
  "Imax": {"value": 28000, "unit": "A"}
}
```

### Step 7: Create WaterFlow Object
- **This is where waterflow_factory is used!**
- Calls `from_flow_params(flow_params)` to create the WaterFlow object

### Step 8: Perform Calculations
- Demonstrates using the WaterFlow object to calculate:
  - Pump speed at various currents
  - Flow rates
  - Pressures
  - Velocities

### Steps 9-10: Save and Visualize
- Creates plots showing experimental data vs fitted curves
- Saves flow_params to JSON file
- Generates `flow_params_pipeline_results.png`

## Output Files

1. **flow_params_output.json** - Saved flow parameters (can be loaded with `WaterFlow.from_file()`)
2. **flow_params_pipeline_results.png** - Visualization of fits

## Key Functions

### `perform_fit(x_data, y_data, fit_function, param_names, quantity_name)`
Core fitting function that:
- Uses `scipy.optimize.curve_fit()` for non-linear least squares
- Calculates parameter uncertainties
- Computes R² fit quality metric

### `fit_pump_speed(data)` → (Vpmax, Vp0)
Fits: `Vp(I) = Vpmax·(I/Imax)² + Vp0`

### `fit_flow_rate(data, vpmax, vp0)` → (F0, Fmax)
Fits: `F(I) = F0 + Fmax·Vp(I)/(Vpmax + Vp0)`

### `fit_pressure(data, vpmax, vp0)` → (Pmin, Pmax)
Fits: `P(I) = Pmin + Pmax·[Vp(I)/(Vpmax + Vp0)]²`

### `create_waterflow_object(flow_params)` → WaterFlow
**Uses the waterflow_factory module to create WaterFlow object**

## Example Output

```
======================================================================
STEP 1: Load Experimental Data from Database Records
======================================================================
(Using synthetic data for demonstration)
  Loaded 500 measurement points
  Current range: 5003 - 27997 A
  Rpm range: 1029 - 3837 rpm
  Flow range: 18.8 - 139.9 l/s
  Pressure range: 5.1 - 25.9 bar

======================================================================
STEP 2: Fit Pump Speed Curve
======================================================================
Model: Vp(I) = Vpmax·(I/Imax)² + Vp0

Pump Speed Fit Results:
  Vpmax = 2839.8543 ± 3.2145
  Vp0 = 1000.1234 ± 1.5678
  R² = 0.999876

...

======================================================================
STEP 7: Create WaterFlow Object using waterflow_factory
======================================================================
WaterFlow object created successfully!
  Type: <class 'python_magnetcooling.waterflow.WaterFlow'>
  Pump speed range: 1000.1234 - 2839.8543 rpm
  Flow range: -0.0123 - 140.0567 l/s
  ...

======================================================================
STEP 8: Perform Hydraulic Calculations
======================================================================
   Current   Pump Speed    Flow Rate   Pressure   Velocity
       [A]        [rpm]       [m³/s]      [bar]      [m/s]
----------------------------------------------------------------------
     10000      1360.45     0.049823      6.45       4.98
     15000      1914.68     0.069871      8.87       6.99
     20000      2610.23     0.095234     12.45       9.52
     ...
```

## Comparison with Real Workflow

### Real `compute()` in flow_params.py:
1. Connects to database API
2. Downloads experimental CSV files
3. Filters and cleans data
4. Performs same curve fits
5. Saves flow_params JSON

### This Pipeline Script:
1. Generates synthetic data (skips database)
2. Performs same curve fits
3. **Uses waterflow_factory module (new!)**
4. Demonstrates calculations
5. Provides visualization

## Integration with Real Data

To use with real database records, replace `generate_synthetic_data()` with:

```python
def load_from_database(session, api_server, headers, oid):
    """Load real experimental data from database"""
    # Use existing code from flow_params.py
    records = utils.get_history(...)
    files = [utils.download(...) for record in records]
    
    # Parse CSV files into arrays
    currents = []
    rpms = []
    flows = []
    pressures = []
    
    for file in files:
        df = pd.read_csv(file, ...)
        currents.extend(df[Ikey].values)
        rpms.extend(df["Rpm"].values)
        # ... etc
    
    return {
        "current": np.array(currents),
        "rpm": np.array(rpms),
        "flow": np.array(flows),
        "pressure": np.array(pressures),
        "back_pressure": np.array(back_pressures),
        "Imax": detected_imax
    }
```

## Dependencies

- numpy
- scipy
- matplotlib
- python_magnetcooling (local package)

## See Also

- `waterflow_factory.py` - Factory functions for creating WaterFlow objects
- `waterflow.py` - WaterFlow class with hydraulic calculation methods
- `flow_params.py` - Real database integration (complex version)
- `waterflow_factory_example.py` - Simple factory usage examples
