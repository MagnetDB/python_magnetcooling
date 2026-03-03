# Hysteresis Model for Water Flow Control (debitbrut)

## Overview

The `debitbrut` method implements a multi-level hysteresis model for computing gross water flow rate as a function of magnet power. This model accounts for the operational reality that flow rates are adjusted based on both current power level AND the direction of power change.

## Hysteresis Behavior

In real magnet cooling systems:
- When **increasing power**: Operators set flow rates conservatively high to ensure adequate cooling
- When **decreasing power**: Flow rates can be reduced, but typically remain higher than the minimum needed
- This creates **hysteresis loops** where the flow rate at a given power depends on whether power is increasing or decreasing

## Parameters

The hysteresis model requires three parameter lists: ### 1. `hysteresis_thresholds` (List[Tuple[float, float]])
List of (ascending_threshold, descending_threshold) pairs for each level (in MW).

Each tuple defines:
- **ascending_threshold**: Power level where flow increases when power is rising
- **descending_threshold**: Power level where flow decreases when power is falling

**Must be ordered from lowest to highest level**
**Descending threshold must be < ascending threshold for each pair**

Example: `[(3, 2), (8, 6), (12, 10)]` means:
- Level 0→1: Increase flow at 3 MW (rising), decrease at 2 MW (falling)
- Level 1→2: Increase flow at 8 MW (rising), decrease at 6 MW (falling)
- Level 2→3: Increase flow at 12 MW (rising), decrease at 10 MW (falling)

### 2. `hysteresis_low_values` (List[float])
Flow rates (in m³/h) for the low state at each level.

**Must have same length as thresholds**

Example: `[100, 200, 300, 400]` means:
- Level 0 low state: 100 m³/h
- Level 1 low state: 200 m³/h
- Level 2 low state: 300 m³/h
- Level 3 low state: 400 m³/h

### 3. `hysteresis_high_values` (List[float])
Flow rates (in m³/h) for the high state at each level.

**Must have same length as thresholds**

Example: `[100, 250, 350, 450]` means:
- Level 0 high state: 100 m³/h
- Level 1 high state: 250 m³/h
- Level 2 high state: 350 m³/h
- Level 3 high state: 450 m³/h

## Usage

### In Python Code

```python
from python_magnetcooling.waterflow import WaterFlow
import numpy as np

# Create WaterFlow with hysteresis parameters
# Each threshold is a tuple: (ascending, descending)
flow = WaterFlow(
    # ... standard parameters ...
    hysteresis_thresholds=[(3, 2), (8, 6), (12, 10)],
    hysteresis_low_values=[100, 200, 300, 400],
    hysteresis_high_values=[100, 250, 350, 450]
)

# Compute flow rate for a power sequence
power_cycle = np.array([0, 5, 10, 15, 10, 5, 0])
flow_rates = flow.debitbrut(power_cycle)
```

### From JSON Configuration

```json
{
  "Vp0": {"value": 1000, "unit": "rpm"},
  "Vpmax": {"value": 2840, "unit": "rpm"},
  "F0": {"value": 0, "unit": "l/s"},
  "Fmax": {"value": 61.7, "unit": "l/s"},
  "Pmax": {"value": 22, "unit": "bar"},
  "Pmin": {"value": 4, "unit": "bar"},
  "BP": {"value": 4, "unit": "bar"},
  "Imax": {"value": 28000, "unit": "A"},
  "hysteresis": {
    "thresholds": [5, 10, 15],
    "low_values": [100, 200, 300, 400],
    "high_values": [100, 250, 350, 450],
    "unit_thresholds": "MW",
    "unit_values": "m³/h"
  }[3, 2], [8, 6], [12, 10]],
    "low_values": [100, 200, 300, 400],
    "high_values": [100, 250, 350, 450],
    "unit_thresholds": "MW (ascending, descending)
Load it:
```python
flow = WaterFlow.from_file("flow_config.json")
```

## Determining Parameters from Data

You can use the functions from `examples/hysteresis.py` to automatically estimate hysteresis parameters from recorded data:

```python
import pandas as pd
import sys
sys.path.insert(0, 'examples')  # Add examples to path
from hysteresis import estimate_hysteresis_parameters, remove_low_x_outliers

# Load your data with columns for power and flow rate
df = pd.read_csv("your_data.csv")  # Must have Pmagnet, debitbrut columns

# Clean outliers from low power region (optional but recommended)
df_clean = remove_low_x_outliers(
    df,
    x_col="Pmagnet",
    y_col="debitbrut",
    x_percentile=25,
    verbose=True
)

# Estimate parameters with clustering for n levels
result = estimate_hysteresis_parameters(
    df_clean,
    x_col="Pmagnet",
    y_col="debitbrut",
    n_levels=3,  # Number of threshold levels
    verbose=True
)

# Extract estimated parameters
thresholds = result["thresholds"]  # List of (asc, desc) tuples
low_values = result["low_values"]  # List of floats
high_values = result["high_values"]  # List of floats

print(f"Thresholds: {thresholds}")
print(f"Low values: {low_values}")
print(f"High values: {high_values}")

# Create WaterFlow with estimated parameters
from python_magnetcooling.waterflow import WaterFlow

flow = WaterFlow(
    hysteresis_thresholds=thresholds,
    hysteresis_low_values=low_values,
    hysteresis_high_values=high_values
)

# Test the model
power_test = df_clean["Pmagnet"].to_numpy()
flow_predicted = flow.debitbrut(power_test)
```

**Note:** The `estimate_hysteresis_parameters` function from `examples/hysteresis.py` already returns thresholds in the correct format as list of `(ascending, descending)` tuples.

## Example Hysteresis Curve

For the parameters above, a power cycle produces:

```
Thresholds: [(3, 2), (8, 6), (12, 10)]
Low values: [100, 200, 300, 400]
High values: [100, 250, 350, 450]

Power Cycle: 0 → 5 → 10 → 15 → 10 → 5 → 0 MW

Power [MW]  Direction      Threshold    Flow [m³/h]
    0       start          -              100 (level 0 low)
    5       increasing     > 3            250 (level 1 high)
   10       increasing     > 8            350 (level 2 high)
   15       increasing     > 12           450 (level 3 high)
   10       decreasing     < 10           350 (level 2 high, stays)
    5       decreasing     < 6            200 (level 1 low)
    0       decreasing     < 2            100 (level 0 low)
```

Notice how at 10 MW:
- When increasing past 8: jumps to 350 m³/h (level 2 high)
- When decreasing from 15: stays at 350 m³/h until drops below 10 MW

This is the **hysteresis effect** - different thresholds for ascending vs descending.

## Validation

The model validates that:
1. `len(low_values) == len(thresholds) + 1`
2. `len(high_values) == len(thresholds) + 1`
3. All parameters are provided before calling `debitbrut()`

## Integration with flow_params_magnetrun.py

The original `debitbrut` function in `examples/flow_params_magnetrun.py` provides a complete pipeline for:
1. Loading magnet run data
2. Cleaning outliers
3. Estimating hysteresis parameters automatically
4. Visualizing the results

However, it requires the `python_magnetrun` package. For a standalone parameter estimation tool, use the functions in `examples/hysteresis.py` instead, which have the same core functionality without external dependencies.

The WaterFlow implementation uses the multi-level hysteresis model from `examples/hysteresis.py`, providing:
- Standalone inference without `python_magnetrun` dependency
- JSON serialization for parameter storage
- Integration with other WaterFlow methods
- Simplified API for power-to-flow conversion

- `flow_rate(current)` - Compute flow from current (no hysteresis)
- `debitbrut(power)` - Compute flow from power (with hysteresis)
- `pressure(current)` - Compute pressure from current
- `velocity(current, cross_section)` - Compute velocity

## References

- Original implementation: `examples/flow_params_magnetrun.py::debitbrut()`
- Hysteresis fitting: `python_magnetrun.processing.hysteresis` module
- Example usage: `examples/waterflow_debitbrut_example.py`
