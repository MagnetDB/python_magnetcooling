# Task: Implement `python_magnetcooling` Package

## Context

We are creating a standalone `python_magnetcooling` package that provides thermal-hydraulic calculations for water-cooled high-field magnets. This package will extract and consolidate cooling-related functionality currently embedded in `python_magnetworkflows`, making it:

- **Reusable** - Can be used in multiple projects (workflows, design tools, post-processing)
- **Standalone** - Independent of FeelPP or any FEM framework
- **Well-tested** - Comprehensive test coverage for all calculations
- **Well-documented** - Clear API with examples for common use cases

## Objectives

1. Extract thermal-hydraulic calculation logic from `python_magnetworkflows`
2. Consolidate all cooling-related modules (waterflow, correlations, friction)
3. Create clean, standalone API for thermal-hydraulic analysis
4. Provide integration adapter for FeelPP workflows
5. Enable use in design tools, optimization, and validation studies

## Package Structure

```
python_magnetcooling/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── setup.py (optional, for backward compatibility)
│
├── tests/
│   ├── __init__.py
│   ├── test_water_properties.py
│   ├── test_correlations.py
│   ├── test_friction.py
│   ├── test_waterflow.py
│   ├── test_channel.py
│   ├── test_thermal_hydraulics.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── flow_params.json
│       └── test_data.csv
│
├── examples/
│   ├── 01_single_channel.py
│   ├── 02_multi_channel_bitter.py
│   ├── 03_helix_design.py
│   ├── 04_waterflow_curves.py
│   ├── 05_axial_discretization.py
│   ├── 06_validation_experiment.py
│   └── 07_feelpp_integration.py
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── quickstart.rst
│   ├── installation.rst
│   ├── api/
│   │   ├── index.rst
│   │   ├── thermal_hydraulics.rst
│   │   ├── waterflow.rst
│   │   ├── correlations.rst
│   │   └── friction.rst
│   ├── theory/
│   │   ├── heat_transfer.rst
│   │   ├── pressure_drop.rst
│   │   └── correlations.rst
│   ├── examples/
│   │   └── gallery/
│   └── migration.rst
│
└── python_magnetcooling/
    ├── __init__.py
    ├── version.py
    ├── exceptions.py           # Custom exceptions
    ├── water_properties.py     # Water/steam properties (IAPWS-IF97)
    ├── correlations.py         # Heat transfer correlations
    ├── friction.py             # Friction factor models
    ├── waterflow.py            # Pump characteristics
    ├── channel.py              # Channel geometry and data structures
    ├── thermal_hydraulics.py  # Main TH calculator
    ├── validators.py           # Input validation
    ├── utils.py                # Utility functions
    └── feelpp/
        ├── __init__.py
        └── adapter.py          # FeelPP integration adapter
```

## Core Modules to Implement

### 1. `exceptions.py` - Custom Exceptions

```python
class MagnetCoolingError(Exception):
    """Base exception for python_magnetcooling"""
    pass

class WaterPropertiesError(MagnetCoolingError):
    """Error in water properties calculation"""
    pass

class CorrelationError(MagnetCoolingError):
    """Error in heat transfer correlation"""
    pass

class FrictionError(MagnetCoolingError):
    """Error in friction factor calculation"""
    pass

class ValidationError(MagnetCoolingError):
    """Input validation error"""
    pass

class ConvergenceError(MagnetCoolingError):
    """Iterative solver did not converge"""
    pass
```

### 2. `water_properties.py` - Water Properties

Extract and refactor water property calculations:

**Current location:** `python_magnetworkflows/cooling.py` (steam, rho, Cp, viscosity, k functions)

**New design:**
```python
from typing import NamedTuple
from iapws import IAPWS97

class WaterState(NamedTuple):
    temperature: float  # K
    pressure: float  # bar
    density: float  # kg/m³
    specific_heat: float  # J/kg/K
    thermal_conductivity: float  # W/m/K
    dynamic_viscosity: float  # Pa·s
    prandtl: float  # dimensionless

class WaterProperties:
    @staticmethod
    def get_state(temperature: float, pressure: float) -> WaterState:
        """Get complete water state at given conditions"""
        # Implementation
    
    @staticmethod
    def compute_temperature_rise(flow_rate, power, temperature, pressure):
        """Compute ΔT = Q / (ρ·cp·V̇)"""
        # Implementation
    
    @staticmethod
    def compute_reynolds(velocity, hydraulic_diameter, temperature, pressure):
        """Compute Reynolds number"""
        # Implementation
```

**Extraction notes:**
- Current `steam()` function → `WaterProperties.get_state()`
- Current `getDT()` function → `WaterProperties.compute_temperature_rise()`
- Use `python_magnetunits` for unit conversions

### 3. `correlations.py` - Heat Transfer Correlations

Extract and refactor heat transfer correlations:

**Current location:** `python_magnetworkflows/cooling.py` (Montgomery, Dittus, Colburn, Silverberg)

**New design:**
```python
from abc import ABC, abstractmethod

class HeatCorrelation(ABC):
    """Base class for heat transfer correlations"""
    
    def __init__(self, fuzzy_factor: float = 1.0):
        self.fuzzy_factor = fuzzy_factor
    
    @abstractmethod
    def compute(self, temperature, pressure, velocity, 
                hydraulic_diameter, length) -> float:
        """Compute heat transfer coefficient [W/m²/K]"""
        pass

class MontgomeryCorrelation(HeatCorrelation):
    """Montgomery correlation for high heat flux"""
    # Implementation

class DittusBoelterCorrelation(HeatCorrelation):
    """Dittus-Boelter correlation"""
    # Implementation

class ColburnCorrelation(HeatCorrelation):
    """Colburn correlation"""
    # Implementation

class SilverbergCorrelation(HeatCorrelation):
    """Silverberg correlation"""
    # Implementation

def get_correlation(name: str, fuzzy_factor: float = 1.0) -> HeatCorrelation:
    """Factory function to get correlation by name"""
    # Implementation

def available_correlations() -> list[str]:
    """Get list of available correlations"""
    # Implementation
```

**Extraction notes:**
- Current functions → class-based design
- Add factory pattern for easy selection
- Keep fuzzy factor support (especially for bitter magnets)

### 4. `friction.py` - Friction Factor Models

Extract and refactor friction factor calculations:

**Current location:** `python_magnetworkflows/cooling.py` (Constant, Blasius, Filonenko, Colebrook, Swamee)

**New design:**
```python
from abc import ABC, abstractmethod

class FrictionModel(ABC):
    """Base class for friction factor models"""
    
    def __init__(self, roughness: float = 0.012e-3):
        self.roughness = roughness
    
    @abstractmethod
    def compute(self, reynolds: float, hydraulic_diameter: float,
                friction_guess: float = 0.055) -> float:
        """Compute friction factor [dimensionless]"""
        pass

class ConstantFriction(FrictionModel):
    """Constant friction factor"""
    # Implementation

class BlasiusFriction(FrictionModel):
    """Blasius correlation for smooth pipes"""
    # Implementation

class FilonenkoFriction(FrictionModel):
    """Filonenko correlation"""
    # Implementation

class ColebrookFriction(FrictionModel):
    """Colebrook-White equation (implicit)"""
    # Implementation

class SwameeFriction(FrictionModel):
    """Swamee-Jain equation (explicit)"""
    # Implementation

def get_friction_model(name: str, roughness: float = 0.012e-3) -> FrictionModel:
    """Factory function"""
    # Implementation
```

**Extraction notes:**
- Current functions → class-based design
- Support for surface roughness
- Iterative solvers for implicit correlations

### 5. `waterflow.py` - Pump Characteristics

Extract and refactor waterflow module:

**Current location:** `python_magnetworkflows/waterflow.py`

**New design:**
```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class WaterFlow:
    """Water flow system characteristics"""
    
    pump_speed_min: float = 1000  # rpm
    pump_speed_max: float = 2840  # rpm
    flow_min: float = 0  # l/s
    flow_max: float = 140  # l/s
    pressure_max: float = 22  # bar
    pressure_min: float = 4  # bar
    pressure_back: float = 4  # bar
    current_max: float = 28000  # A
    
    @classmethod
    def from_file(cls, filename: str) -> "WaterFlow":
        """Load from JSON file"""
        # Implementation
    
    def pump_speed(self, current: float) -> float:
        """Compute pump speed as function of current"""
        # Implementation
    
    def flow_rate(self, current: float) -> float:
        """Compute flow rate [m³/s]"""
        # Implementation
    
    def pressure(self, current: float) -> float:
        """Compute system pressure [bar]"""
        # Implementation
    
    def pressure_drop(self, current: float) -> float:
        """Compute pressure drop [bar]"""
        # Implementation
    
    def velocity(self, current: float, cross_section: float) -> float:
        """Compute mean velocity [m/s]"""
        # Implementation
    
    def to_file(self, filename: str) -> None:
        """Save to JSON file"""
        # Implementation
```

**Extraction notes:**
- Refactor from class `waterflow` to `WaterFlow` (PEP 8)
- Use `python_magnetunits` for conversions
- Add validation
- Keep JSON file format compatibility

### 6. `channel.py` - Channel Geometry and Data Structures

**New module** - Define data structures:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ChannelGeometry:
    """Geometric parameters for a cooling channel"""
    hydraulic_diameter: float  # m
    cross_section: float  # m²
    length: float  # m
    name: str = ""

@dataclass
class AxialDiscretization:
    """Axial discretization for gradHZ mode"""
    z_positions: List[float]  # m
    power_distribution: List[float]  # W

@dataclass
class ChannelInput:
    """Input parameters for thermal-hydraulic calculation"""
    geometry: ChannelGeometry
    power: float  # W
    temp_inlet: float  # K
    axial_discretization: Optional[AxialDiscretization] = None
    # Initial guesses
    temp_outlet_guess: Optional[float] = None
    heat_coeff_guess: Optional[float] = None
    velocity_guess: Optional[float] = None

@dataclass
class ChannelOutput:
    """Results from thermal-hydraulic calculation"""
    velocity: float  # m/s
    flow_rate: float  # m³/s
    friction_factor: float
    temp_inlet: float  # K
    temp_outlet: float  # K
    temp_rise: float  # K
    temp_mean: float  # K
    heat_coeff: float  # W/m²/K
    heat_coeff_distribution: Optional[List[float]] = None
    temp_distribution: Optional[List[float]] = None
    density_outlet: float = 0.0  # kg/m³
    specific_heat_outlet: float = 0.0  # J/kg/K
    converged: bool = True
    iterations: int = 0
```

### 7. `thermal_hydraulics.py` - Main Calculator

**New module** - Main thermal-hydraulic solver:

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ThermalHydraulicInput:
    """Complete input for thermal-hydraulic analysis"""
    channels: List[ChannelInput]
    pressure_inlet: float  # bar
    pressure_drop: float  # bar
    heat_correlation: str = "Montgomery"
    friction_model: str = "Constant"
    fuzzy_factor: float = 1.0
    extra_pressure_loss: float = 1.0
    max_iterations: int = 10
    tolerance_flow: float = 1e-3
    tolerance_temp: float = 1e-3
    relaxation_factor: float = 0.0

@dataclass
class ThermalHydraulicOutput:
    """Complete thermal-hydraulic analysis results"""
    channels: List[ChannelOutput]
    total_flow_rate: float  # m³/s
    outlet_temp_mixed: float  # K
    total_power: float  # W
    max_error_temp: float = 0.0
    max_error_heat_coeff: float = 0.0
    converged: bool = True

class ThermalHydraulicCalculator:
    """
    Standalone thermal-hydraulic calculator
    
    Can be used independently for:
    - Design calculations
    - Sensitivity studies
    - Optimization
    - Validation against experimental data
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def compute(self, inputs: ThermalHydraulicInput) -> ThermalHydraulicOutput:
        """Compute thermal-hydraulic solution"""
        # Implementation
    
    def compute_from_waterflow(self, inputs: ThermalHydraulicInput,
                               waterflow_params: WaterFlow,
                               current: float) -> ThermalHydraulicOutput:
        """Compute using waterflow pump characteristics"""
        # Implementation
    
    def _compute_channel_uniform(self, channel, global_inputs):
        """Compute channel with uniform properties"""
        # Implementation
    
    def _compute_channel_axial(self, channel, global_inputs):
        """Compute channel with axial discretization"""
        # Implementation
    
    def _compute_mixed_outlet_temp(self, channels, pressure):
        """Compute mixed outlet temperature"""
        # Implementation

def compute_single_channel(hydraulic_diameter, cross_section, length,
                          power, temp_inlet, pressure_inlet, pressure_drop,
                          **kwargs) -> ChannelOutput:
    """Convenience function for single channel calculation"""
    # Implementation
```

**Extraction notes:**
- Extract logic from `python_magnetworkflows/error.py` (compute_error function)
- Extract logic from `python_magnetworkflows/cooling.py` (Uw, getHeatCoeff, getTout)
- Separate iterative solver logic from FeelPP integration
- Support both uniform and axial discretization modes

### 8. `feelpp/adapter.py` - FeelPP Integration Adapter

**New module** - Bridge to FeelPP workflows:

```python
class FeelppThermalHydraulicAdapter:
    """
    Adapter to use standalone calculator with FeelPP data structures
    
    Converts between:
    - FeelPP parameter dictionaries → ThermalHydraulicInput
    - ThermalHydraulicOutput → FeelPP parameter updates
    """
    
    def __init__(self, calculator: ThermalHydraulicCalculator):
        self.calculator = calculator
    
    def compute_from_feelpp_data(self, target, dict_df, p_params,
                                 parameters, targets, args, basedir):
        """Compute from FeelPP data structures"""
        # Build input
        th_input = self._build_input_from_feelpp(...)
        
        # Compute
        th_output = self.calculator.compute(th_input)
        
        # Convert back
        parameters_update = self._extract_parameter_updates(...)
        dict_df_update = self._update_dict_df(...)
        
        return th_output, parameters_update, dict_df_update
    
    def _build_input_from_feelpp(self, ...):
        """Build ThermalHydraulicInput from FeelPP data"""
        # Implementation
    
    def _extract_parameter_updates(self, ...):
        """Extract parameter updates for FeelPP"""
        # Implementation
    
    def _update_dict_df(self, ...):
        """Update dict_df with results"""
        # Implementation
```

## Dependencies

### `pyproject.toml`

```toml
[project]
name = "python_magnetcooling"
version = "0.1.0"
description = "Thermal-hydraulic calculations for water-cooled high-field magnets"
dependencies = [
    "numpy>=2.0.0",
    "scipy>=1.14.0",
    "pandas>=2.2.0",
    "iapws>=1.4.0",
    "python_magnetunits>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "mypy>=1.8.0",
]
feelpp = [
    "feelpp>=0.102.1",
    "feelpp-toolboxes>=0.102.1",
]
viz = [
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

1. **Setup package structure**
   - Create repository
   - Set up pyproject.toml
   - Configure testing framework
   - Set up CI/CD

2. **Implement core modules**
   - `exceptions.py` - Custom exceptions
   - `water_properties.py` - Extract from cooling.py
   - `channel.py` - Define data structures

3. **Basic tests**
   - Test water properties
   - Test data structure validation

### Phase 2: Correlations and Models (Week 2-3)

1. **Implement correlation modules**
   - `correlations.py` - Extract and refactor heat transfer correlations
   - `friction.py` - Extract and refactor friction models

2. **Implement waterflow**
   - `waterflow.py` - Refactor existing module
   - Add validation
   - Integrate with python_magnetunits

3. **Comprehensive tests**
   - Test all correlations
   - Test friction models
   - Test waterflow calculations
   - Compare with existing implementation

### Phase 3: Main Calculator (Week 3-4)

1. **Implement thermal_hydraulics.py**
   - Extract logic from error.py and cooling.py
   - Implement uniform channel calculation
   - Implement axial discretization
   - Implement convergence logic

2. **Implement convenience functions**
   - `compute_single_channel()`
   - Helper utilities

3. **Integration tests**
   - Test complete workflows
   - Test convergence behavior
   - Test error handling

### Phase 4: FeelPP Integration (Week 4-5)

1. **Implement FeelPP adapter**
   - `feelpp/adapter.py`
   - Data structure conversion
   - Parameter extraction

2. **Integration with python_magnetworkflows**
   - Update error.py to use adapter
   - Verify compatibility
   - Test with real simulations

3. **Migration documentation**
   - Document API changes
   - Provide migration examples
   - Update python_magnetworkflows docs

### Phase 5: Documentation and Examples (Week 5-6)

1. **Complete documentation**
   - API documentation (Sphinx)
   - Theory documentation
   - User guide

2. **Create examples**
   - Single channel calculation
   - Multi-channel systems
   - Waterflow curves
   - Axial discretization
   - FeelPP integration
   - Validation against experiments

3. **Package release**
   - PyPI upload
   - Debian package
   - Announce to users

## Testing Strategy

### Unit Tests

```python
# tests/test_water_properties.py
def test_water_properties_valid_range():
    state = WaterProperties.get_state(300.0, 15.0)
    assert state.density > 0
    assert state.specific_heat > 0

def test_water_properties_invalid_temperature():
    with pytest.raises(WaterPropertiesError):
        WaterProperties.get_state(100.0, 15.0)  # Too cold

# tests/test_correlations.py
def test_montgomery_correlation():
    corr = MontgomeryCorrelation(fuzzy_factor=1.0)
    h = corr.compute(300.0, 15.0, 5.0, 0.008, 0.5)
    assert h > 0
    assert h < 1e6  # Reasonable range

def test_dittus_correlation_laminar_warning():
    corr = DittusBoelterCorrelation()
    with pytest.raises(CorrelationError):
        corr.compute(300.0, 15.0, 0.1, 0.008, 0.5)  # Too low Re

# tests/test_thermal_hydraulics.py
def test_single_channel_calculation():
    result = compute_single_channel(
        hydraulic_diameter=0.008,
        cross_section=5e-5,
        length=0.5,
        power=50000,
        temp_inlet=290.0,
        pressure_inlet=15.0,
        pressure_drop=5.0
    )
    assert result.converged
    assert result.temp_outlet > result.temp_inlet
    assert result.velocity > 0
```

### Integration Tests

```python
# tests/test_integration.py
def test_with_waterflow():
    """Test integration with WaterFlow"""
    flow = WaterFlow.from_file("tests/fixtures/flow_params.json")
    
    # Create channel
    channel = ChannelInput(...)
    
    # Create inputs
    inputs = ThermalHydraulicInput(channels=[channel], ...)
    
    # Compute
    calc = ThermalHydraulicCalculator()
    result = calc.compute_from_waterflow(inputs, flow, current=20000)
    
    assert result.converged
    assert result.total_flow_rate > 0

def test_feelpp_adapter():
    """Test FeelPP adapter"""
    # Mock FeelPP data structures
    dict_df = {...}
    p_params = {...}
    parameters = {...}
    
    # Create adapter
    calc = ThermalHydraulicCalculator()
    adapter = FeelppThermalHydraulicAdapter(calc)
    
    # Compute
    th_output, params_update, df_update = adapter.compute_from_feelpp_data(...)
    
    assert th_output.converged
    assert "hw_Channel1" in params_update
```

### Validation Tests

```python
# tests/test_validation.py
def test_against_experimental_data():
    """Validate against known experimental results"""
    # Load experimental data
    exp_data = pd.read_csv("tests/fixtures/experiment.csv")
    
    # Run simulation
    result = compute_single_channel(...)
    
    # Compare
    assert abs(result.temp_outlet - exp_data["Tout"]) < 5.0  # Within 5K
```

### Regression Tests

```python
# tests/test_regression.py
def test_compatibility_with_old_implementation():
    """Ensure results match old python_magnetworkflows implementation"""
    # This test can be removed after migration is complete
    
    # Old implementation (from python_magnetworkflows)
    old_result = old_compute_function(...)
    
    # New implementation
    new_result = compute_single_channel(...)
    
    # Should match within tolerance
    assert abs(new_result.temp_outlet - old_result) < 1.0
```

## Documentation Requirements

### API Documentation

- Auto-generated from docstrings
- Type hints for all public functions
- Mathematical formulas in docstrings
- Examples for each major function

### User Guide

```rst
Quickstart
----------

Installation::

    pip install python_magnetcooling

Basic usage::

    from python_magnetcooling import compute_single_channel
    
    result = compute_single_channel(
        hydraulic_diameter=0.008,  # 8 mm
        cross_section=5e-5,        # 50 mm²
        length=0.5,                # 50 cm
        power=50000,               # 50 kW
        temp_inlet=290.0,          # K
        pressure_inlet=15.0,       # bar
        pressure_drop=5.0          # bar
    )
    
    print(f"Outlet temperature: {result.temp_outlet:.2f} K")
    print(f"Heat coefficient: {result.heat_coeff:.0f} W/m²/K")
```

### Theory Documentation

- Heat transfer fundamentals
- Nusselt correlations (with references)
- Friction factor models (with references)
- Pressure drop calculations
- Axial discretization method

### Examples Gallery

- Single channel calculation
- Multi-channel bitter magnet
- Multi-channel helix magnet
- Waterflow pump curves
- Axial power distribution (gradHZ)
- Parameter sensitivity study
- Optimization example
- Validation against experiment
- Integration with FeelPP

## Integration with python_magnetworkflows

### Update Dependencies

```toml
# python_magnetworkflows/pyproject.toml
[project]
dependencies = [
    "python_magnetunits>=0.1.0",
    "python_magnetcooling>=0.1.0",
    # ... other dependencies
]
```

### Update error.py

```python
# python_magnetworkflows/error.py
from python_magnetcooling.feelpp import FeelppThermalHydraulicAdapter
from python_magnetcooling import ThermalHydraulicCalculator

def compute_error(e, f, basedir, it, args, targets, postvalues, params, parameters):
    """Compute error using python_magnetcooling"""
    
    # Create calculator
    calc = ThermalHydraulicCalculator(verbose=args.debug)
    adapter = FeelppThermalHydraulicAdapter(calc)
    
    # Initialize dict_df
    dict_df = init_dict_df(targets, args)
    
    # ... existing code to extract parameters ...
    
    for target, values in targets.items:
        # Compute thermal-hydraulics using adapter
        th_output, params_update, df_update = adapter.compute_from_feelpp_data(
            target, dict_df, p_params, parameters, targets, args, basedir
        )
        
        # Update parameters
        parameters.update(params_update)
        
        # Update dict_df
        dict_df[target].update(df_update)
        
        # Extract errors
        err_max_dT = max(err_max_dT, th_output.max_error_temp)
        err_max_h = max(err_max_h, th_output.max_error_heat_coeff)
    
    # ... rest of function ...
```

### Migration Path

1. **Phase 1**: Install python_magnetcooling alongside existing code
2. **Phase 2**: Update error.py to use adapter
3. **Phase 3**: Remove old cooling.py functions
4. **Phase 4**: Update tests
5. **Phase 5**: Update documentation

## Success Criteria

### Functional Requirements

- ✓ All correlations produce correct results
- ✓ Convergence behavior matches existing implementation
- ✓ Results match experimental data within acceptable tolerance
- ✓ FeelPP integration works seamlessly
- ✓ All examples run without errors

### Quality Requirements

- ✓ Test coverage > 90%
- ✓ All public functions documented
- ✓ Type hints throughout
- ✓ No breaking changes to python_magnetworkflows
- ✓ Performance comparable to existing implementation

### Usability Requirements

- ✓ Clear error messages
- ✓ Comprehensive examples
- ✓ Easy to install (pip, Debian)
- ✓ Works standalone (no FeelPP required)
- ✓ Works with FeelPP (via adapter)

## Deliverables

1. **Package**
   - Source code on GitHub
   - PyPI package
   - Debian package

2. **Documentation**
   - API documentation (HTML)
   - User guide
   - Theory documentation
   - Examples gallery
   - Migration guide

3. **Tests**
   - Unit tests (>90% coverage)
   - Integration tests
   - Validation tests
   - Regression tests

4. **Examples**
   - At least 7 working examples
   - Jupyter notebooks (optional)

5. **Updated python_magnetworkflows**
   - Uses python_magnetcooling
   - Updated documentation
   - Passing all tests

## Timeline

- **Week 1-2**: Foundation and core modules
- **Week 3-4**: Main calculator
- **Week 4-5**: FeelPP integration
- **Week 5-6**: Documentation and release
- **Week 7**: Buffer for issues and refinement

**Total: 6-7 weeks**

## Questions to Address During Implementation

### Technical Questions

1. **Convergence criteria**: What are appropriate tolerances?
2. **Performance**: Are there hot paths that need optimization?
3. **Edge cases**: How to handle very low/high Reynolds numbers?
4. **Validation**: What experimental data is available?

### Design Questions

1. **API stability**: Which parts of API should be stable in v1.0?
2. **Extensibility**: How to support custom correlations?
3. **Backward compatibility**: How long to maintain deprecated code?

### Integration Questions

1. **FeelPP versions**: Which versions to support?
2. **Data formats**: Any changes needed to JSON/CSV formats?
3. **Performance**: Any performance impact from abstraction?

## Risk Mitigation

### Risk 1: Breaking Changes

**Mitigation**: 
- Comprehensive tests comparing old and new implementations
- Gradual migration with adapter layer
- Clear deprecation warnings

### Risk 2: Performance Degradation

**Mitigation**:
- Benchmark critical paths
- Profile before and after
- Optimize if needed

### Risk 3: Missing Features

**Mitigation**:
- Careful audit of existing functionality
- Test all use cases from python_magnetworkflows
- Beta testing with users

### Risk 4: Documentation Gaps

**Mitigation**:
- Write docs alongside code
- Include examples for all major features
- User testing of documentation

## Next Steps

1. **Review this prompt** with stakeholders
2. **Set up development environment**
3. **Create initial repository structure**
4. **Begin Phase 1 implementation**
5. **Schedule regular check-ins** to track progress

---

## Appendix: Code to Extract

### From `python_magnetworkflows/cooling.py`

Extract these functions:
- `steam()` → `WaterProperties.get_state()`
- `getDT()` → `WaterProperties.compute_temperature_rise()`
- `getHeatCoeff()` → Use correlation classes
- `getTout()` → `ThermalHydraulicCalculator._compute_mixed_outlet_temp()`
- `Montgomery()` → `MontgomeryCorrelation.compute()`
- `Dittus()` → `DittusBoelterCorrelation.compute()`
- `Colburn()` → `ColburnCorrelation.compute()`
- `Silverberg()` → `SilverbergCorrelation.compute()`
- `Constant()` → `ConstantFriction.compute()`
- `Blasius()` → `BlasiusFriction.compute()`
- `Filonenko()` → `FilonenkoFriction.compute()`
- `Colebrook()` → `ColebrookFriction.compute()`
- `Swamee()` → `SwameeFriction.compute()`
- `Uw()` → Internal to `ThermalHydraulicCalculator`
- `Reynolds()` → `WaterProperties.compute_reynolds()`
- `Prandlt()` → Computed in `WaterState`

### From `python_magnetworkflows/waterflow.py`

Refactor entire file:
- Class `waterflow` → `WaterFlow`
- All methods remain but with better typing
- Add validation
- Use python_magnetunits

### From `python_magnetworkflows/error.py`

Extract thermal-hydraulic calculation logic:
- Channel-by-channel calculation loop
- Axial discretization handling (FluxZ)
- Convergence checking
- Mixed outlet temperature calculation

Keep in python_magnetworkflows:
- FeelPP-specific parameter extraction
- Target management
- Result aggregation for FeelPP

---

This prompt should provide complete guidance for implementing the python_magnetcooling package!
