# Python Magnet Cooling

Thermal-hydraulic calculations for water-cooled high-field magnets.

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`python_magnetcooling` is a Python package for performing thermal-hydraulic analysis of water-cooled high-field magnets. It provides comprehensive tools for computing heat transfer, fluid flow, and temperature distributions in cooling channels, with a particular focus on the requirements of high-power resistive magnets.

### Key Features

- **Thermal-Hydraulic Solver**: Complete analysis of cooling channels including temperature rise, heat transfer coefficients, and flow parameters
- **Heat Transfer Correlations**: Support for multiple correlations (Montgomery, Dittus-Boelter, etc.)
- **Water Properties**: IAPWS-IF97 standard water/steam properties via `iapws` library
- **Friction Models**: Various friction factor correlations for turbulent flow
- **Axial Discretization**: Support for non-uniform power distribution along channel length
- **Heat Exchanger Analysis**: Primary cooling loop heat exchanger calculations
- **Unit-Aware Calculations**: Integration with `pint` for physical quantities

## Installation

### Requirements

- Python 3.11 or higher
- NumPy >= 2.0.0
- SciPy >= 1.14.0
- pandas >= 2.2.0
- iapws >= 1.4.0
- pint >= 0.17.1
- ht >= 1.2.0

### Install from Source

```bash
git clone https://github.com/MagnetDB/python_magnetcooling.git
cd python_magnetcooling
pip install -e .
```

### Install within a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Optional Dependencies

For development and testing:
```bash
pip install -e ".[dev]"
```

For documentation:
```bash
pip install -e ".[docs]"
```

For visualization (plotting utilities):
```bash
pip install -e ".[viz]"
# Or manually: pip install matplotlib
```

For Clawpack PDE solvers (required for `clawtest1.py`):
```bash
pip install -e ".[clawpack]"
```

**Note:** Plotting functions in `fitting` and `hysteresis` modules require `matplotlib`. If not installed, plotting functions will display a warning and return gracefully.

## Quick Start

### Single Channel Calculation

```python
from python_magnetcooling import compute_single_channel

# Define channel parameters
result = compute_single_channel(
    hydraulic_diameter=0.008,  # 8 mm
    cross_section=5e-5,        # 50 mm²
    length=0.5,                # 0.5 m
    power=50000,               # 50 kW
    temp_inlet=290.0,          # 290 K (17°C)
    pressure_inlet=15.0,       # 15 bar
    pressure_drop=5.0          # 5 bar
)

print(f"Outlet temperature: {result.temp_outlet:.2f} K")
print(f"Flow velocity: {result.velocity:.2f} m/s")
print(f"Heat transfer coefficient: {result.heat_coeff:.1f} W/m²/K")
```

### Advanced Usage with Axial Discretization

```python
from python_magnetcooling.thermohydraulics import (
    ThermalHydraulicCalculator,
    ChannelInput,
    ChannelGeometry,
    AxialDiscretization
)

# Define geometry
geometry = ChannelGeometry(
    hydraulic_diameter=0.008,
    cross_section=5e-5,
    length=0.5,
    name="Inner helix"
)

# Define non-uniform power distribution
discretization = AxialDiscretization(
    z_positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # m
    power_distribution=[8000, 12000, 15000, 12000, 8000]  # W per section
)

# Create input
channel_input = ChannelInput(
    geometry=geometry,
    power=55000,  # Total power in W
    temp_inlet=290.0,
    axial_discretization=discretization
)

# Run calculation
calculator = ThermalHydraulicCalculator(
    pressure_inlet=15.0,
    pressure_drop=5.0
)
result = calculator.solve_channel(channel_input)

# Access per-section dTw (feelpp reconstructs Tw as T_in + cumsum(dTw))
T_in = result.temp_inlet
for i, dTw in enumerate(result.temp_rise_distribution):
    T_section = T_in + sum(result.temp_rise_distribution[:i])
    print(f"section {i}: dTw = {dTw:.4f} K, T_start = {T_section:.2f} K")
```

## Main Modules

### `thermohydraulics`
Core thermal-hydraulic solver for cooling channels. Handles single-point and axially-discretized calculations.

**Key Classes:**
- `ThermalHydraulicCalculator`: Main solver
- `ChannelGeometry`: Geometric parameters
- `ChannelInput`/`ChannelOutput`: Input/output data structures

### `correlations`
Heat transfer and Nusselt number correlations for forced convection.

**Available Correlations:**
- Montgomery correlation (for magnet cooling)
- Dittus-Boelter correlation
- Sieder-Tate correlation
- Gnielinski correlation

### `water_properties`
Water and steam thermophysical properties based on IAPWS-IF97 standard.

**Functions:**
- Density, viscosity, thermal conductivity
- Specific heat capacity
- Enthalpy and entropy
- Properties as functions of (T, P) or (P, h)

### `friction`
Friction factor models for pressure drop calculations.

**Models:**
- Darcy-Weisbach equation
- Churchill correlation
- Colebrook-White equation
- Smooth and rough pipe models

### `fitting`
Hydraulic system curve fitting for pump speed, flow rate, and pressure relationships.

**Key Features:**
- Pump speed vs current fitting (quadratic and piecewise linear methods)
- Flow rate and pressure curve fitting
- Hysteresis parameter estimation for flow control systems
- Fit quality metrics (RMSE, MAE, R², MAPE) for model evaluation
- Integration with experimental data from magnet runs
- Plotting utilities for visualization

**Main Functions:**
- `fit_hydraulic_system()`: Comprehensive system fitting
- `fit_hysteresis_parameters()`: Fit multi-level hysteresis models
- `compute_pump_fit_metrics()`, `compute_flow_fit_metrics()`, `compute_pressure_fit_metrics()`: Evaluate fit quality
- `compute_all_hydraulic_metrics()`: Get all metrics at once
- `compute_hysteresis_fit_metrics()`: Hysteresis model quality metrics
- `plot_pump_fit()`, `plot_flow_pressure_fit()`, `plot_hysteresis_fit()`: Visualization tools

### `hysteresis`
Multi-level hysteresis models for flow control systems with history-dependent behavior.

**Key Features:**
- Multi-level hysteresis with separate ascending/descending thresholds
- Parameter estimation from time-series data
- Data cleaning and outlier removal utilities
- Fit quality metrics for model evaluation
- Visualization tools for hysteresis loops and fits

**Main Functions:**
- `multi_level_hysteresis()`: Apply hysteresis model to input signal
- `estimate_hysteresis_parameters()`: Estimate parameters from data
- `compute_hysteresis_fit_metrics()`: Evaluate model quality with RMSE, R², match rate
- `remove_outliers()`, `remove_low_x_outliers()`: Data cleaning
- `plot_hysteresis_model()`, `plot_hysteresis_fit()`: Visualization

### `waterflow`
Pump characteristics and flow rate calculations for cooling loops.

**Key Features:**
- Pump speed, flow rate, and pressure calculations as functions of current
- Support for hysteresis models for secondary cooling loop flow (`debitbrut()` method)
- JSON serialization for configuration management

### `heatexchanger_primary`
Analysis of primary cooling loop heat exchangers with temperature field calculations.

### `clawtest1` (requires optional `clawpack` dependency)
One-dimensional advection solver for cooling loop modeling using Clawpack. Solves the linear advection equation for water temperature evolution in the cooling circuit.

**Note:** This module requires the optional `clawpack` dependency. Install with `pip install -e ".[clawpack]"`.

## Examples

The `examples/` directory contains practical applications:

- **`heatexchanger_primary.py`**: Complete heat exchanger analysis with temperature profiles and visualization
- **`feelpp.py`**: Integration with Feel++ finite element simulations
- **`plotting_example.py`**: Demonstrates plotting utilities for hydraulic fits and hysteresis models
- **`fit_metrics_example.py`**: Shows how to evaluate fit quality and decide when to refit models
- **`waterflow_debitbrut_example.py`**: Usage examples for secondary cooling loop flow rate calculation with hysteresis
- **`hysteresis_demo.py`**: Comprehensive hysteresis model demonstrations and parameter estimation

To run an example:
```bash
python examples/heatexchanger_primary.py <input_file> --nhelices 14
python examples/fit_metrics_example.py  # Evaluate fit quality metrics
python examples/plotting_example.py  # Creates visualization plots
```

## Physical Background

This package implements thermal-hydraulic models specifically adapted for high-field resistive magnets:

1. **Forced convection** in circular or non-circular channels
2. **Turbulent flow** regime (Re > 2300 typically)
3. **Single-phase liquid cooling** (water below saturation temperature)
4. **High heat flux** conditions (>10 MW/m²)
5. **Pressure drops** from 1 to 20 bar

The correlations and models have been validated against experimental data from resistive magnet operations at LNCMI.

## Development

### Running Tests

⚠️ **Important:** Before running tests, install the package with development dependencies:

```bash
pytest [--cov=python_magnetcooling --cov-report=html]
```

For detailed testing instructions, see [TESTING.md](TESTING.md).

#### Test Suite Overview

The test suite includes comprehensive tests for:

- **Exception Handling**: All custom exception classes and inheritance
- **Water Properties**: IAPWS-IF97 calculations, state properties, temperature/pressure variations
- **Heat Transfer Correlations**: Nusselt number calculations, Montgomery correlation
- **Friction Models**: Constant, Blasius, and other friction factor correlations
- **Module Imports**: Package structure and basic functionality

See [tests/README.md](tests/README.md) for detailed testing documentation.

### Code Formatting

This project uses `black` for code formatting:
```bash
black python_magnetcooling/
```

### Type Checking

```bash
mypy python_magnetcooling/
```

## Documentation

Full documentation is available at [python-magnetcooling.readthedocs.io](https://python-magnetcooling.readthedocs.io)

To build documentation locally:
```bash
cd docs
make html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Christophe Trophime** - *Initial work* - [LNCMI-CNRS](mailto:christophe.trophime@lncmi.cnrs.fr)

## Acknowledgments

- Developed at Laboratoire National des Champs Magnétiques Intenses (LNCMI), CNRS
- Supports thermal analysis of resistive magnets for scientific research

## Citation

If you use this package in your research, please cite:

```bibtex
@software{python_magnetcooling,
  author = {Trophime, Christophe},
  title = {Python Magnet Cooling: Thermal-hydraulic calculations for water-cooled high-field magnets},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/MagnetDB/python_magnetcooling}
}
```

## Support

For questions and support, please open an issue on the [GitHub repository](https://github.com/MagnetDB/python_magnetcooling/issues).
