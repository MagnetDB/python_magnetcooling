# WaterFlow Factory Module - Extraction Summary

## Overview

The WaterFlow object creation logic has been extracted from the database loading code into a separate, reusable module: `python_magnetcooling/waterflow_factory.py`.

## What Was Created

### 1. New Module: `waterflow_factory.py`

This module provides factory functions to create `WaterFlow` objects from various data sources:

#### Factory Functions

- **`from_flow_params(params: Dict)`**  
  Create WaterFlow from the flow_params dictionary format used by the `compute()` method in `examples/flow_params.py`
  
  ```python
  from python_magnetcooling.waterflow_factory import from_flow_params
  
  params = {
      "Vp0": {"value": 1000, "unit": "rpm"},
      "Vpmax": {"value": 2840, "unit": "rpm"},
      # ... more parameters
  }
  flow = from_flow_params(params)
  ```

- **`from_database_record(record: Dict, key_mapping: Optional[Dict] = None)`**  
  Create WaterFlow from database records with flexible field mapping
  
  ```python
  from python_magnetcooling.waterflow_factory import from_database_record
  
  # Custom database schema
  db_record = {
      "min_pump_rpm": 1000,
      "max_pump_rpm": 2840,
      # ...
  }
  
  mapping = {
      "Vp0": "min_pump_rpm",
      "Vpmax": "max_pump_rpm",
      # ...
  }
  
  flow = from_database_record(db_record, mapping)
  ```

- **`from_fitted_data(pump_speed_fit, flow_rate_fit, pressure_fit, back_pressure, max_current)`**  
  Create WaterFlow directly from fitted curve parameters
  
  ```python
  from python_magnetcooling.waterflow_factory import from_fitted_data
  
  flow = from_fitted_data(
      pump_speed_fit=(2840, 1000),  # (Vpmax, Vp0)
      flow_rate_fit=(0, 140),        # (F0, Fmax)
      pressure_fit=(4, 22),          # (Pmin, Pmax)
      back_pressure=4.0,
      max_current=28000
  )
  ```

- **`create_default()`**  
  Create WaterFlow with default parameter values
  
  ```python
  from python_magnetcooling.waterflow_factory import create_default
  
  flow = create_default()
  ```

### 2. Updated Files

- **`python_magnetcooling/__init__.py`**  
  Added imports and exports for the new factory functions. The functions are now available directly from the main package:
  
  ```python
  from python_magnetcooling import (
      from_flow_params,
      from_database_record,
      from_fitted_data,
      create_default_waterflow,
  )
  ```

### 3. Tests: `tests/test_waterflow_factory.py`

Comprehensive test suite covering all factory functions:
- Creating from flow_params with BP and Pout keys
- Creating from database records with custom mappings
- Creating from fitted data
- Verifying that created WaterFlow objects work correctly

### 4. Example: `examples/waterflow_factory_example.py`

Detailed examples demonstrating:
- How to use each factory function
- Integration with the `compute()` workflow
- Backward compatibility with existing code

## Benefits

### Separation of Concerns
- **Data acquisition/fitting**: `examples/flow_params.py` (compute method)
- **Object creation**: `waterflow_factory.py`
- **Hydraulic calculations**: `WaterFlow` class methods

### Improved Maintainability
- Centralized object creation logic
- Easy to modify without touching database code
- Clear interface for different data sources

### Better Testing
- Factory functions can be tested independently
- Mock data can be easily created for tests
- No database dependency for testing WaterFlow creation

### Flexibility
- Support for different database schemas via key mapping
- Works with fitted data or raw parameters
- Backward compatible with existing `WaterFlow.from_file()`

## Usage in `compute()` Method

The `compute()` method in `examples/flow_params.py` can now optionally use the factory:

```python
def compute(session, api_server, headers, oid, samples=20, debug=False):
    # ... existing code that builds flow_params dictionary ...
    
    # Save parameters as before
    with open(filename, "w") as f:
        f.write(json.dumps(flow_params, indent=4))
    
    # NEW: Create WaterFlow object using factory
    from python_magnetcooling.waterflow_factory import from_flow_params
    waterflow = from_flow_params(flow_params)
    
    # Now you can return the object for immediate use
    return waterflow
```

## Backward Compatibility

The existing `WaterFlow.from_file()` method continues to work exactly as before:

```python
from python_magnetcooling import WaterFlow

# Still works!
flow = WaterFlow.from_file("M9_M10-flow_params.json")
```

The factory provides an alternative, more flexible approach, especially useful when working with in-memory data rather than files.

## Next Steps

### Optional Enhancements

1. **Update `examples/flow_params.py`**  
   Modify the `compute()` function to use the factory and return WaterFlow objects

2. **Add more factory methods**  
   E.g., `from_api_response()`, `from_dataframe()`, etc.

3. **Validation**  
   Add parameter validation in factory functions to catch invalid data early

4. **Documentation**  
   Add examples to main documentation in `docs/`

## Files Created/Modified

### Created:
- `python_magnetcooling/waterflow_factory.py` - Factory functions module
- `tests/test_waterflow_factory.py` - Test suite
- `examples/waterflow_factory_example.py` - Usage examples
- `WATERFLOW_FACTORY_SUMMARY.md` - This document

### Modified:
- `python_magnetcooling/__init__.py` - Added factory function imports and exports

## Running Tests

```bash
# Run all waterflow_factory tests
python -m pytest tests/test_waterflow_factory.py -v

# Run specific test
python -m pytest tests/test_waterflow_factory.py::test_from_flow_params -v
```

## Running Examples

```bash
cd examples
python waterflow_factory_example.py
```

This will demonstrate all factory functions with sample data and show how to integrate them into your workflow.
