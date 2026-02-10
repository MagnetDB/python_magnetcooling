"""
Example: Creating WaterFlow objects from database records using waterflow_factory

This example demonstrates how to use the new waterflow_factory module to create
WaterFlow objects from various data sources, particularly from database records
as loaded by the compute() method in flow_params.py.
"""

from python_magnetcooling.waterflow_factory import (
    from_flow_params,
    from_database_record,
    from_fitted_data,
    create_default,
)


def example_from_flow_params():
    """
    Example 1: Create WaterFlow from flow_params dictionary
    
    This is the most common use case - the compute() method in flow_params.py
    generates a flow_params dictionary that can be directly used here.
    """
    print("=" * 70)
    print("Example 1: Creating WaterFlow from flow_params dictionary")
    print("=" * 70)
    
    # This format matches the output from compute() in examples/flow_params.py
    flow_params = {
        "Vp0": {"value": 1000, "unit": "rpm"},
        "Vpmax": {"value": 2840, "unit": "rpm"},
        "F0": {"value": 0, "unit": "l/s"},
        "Fmax": {"value": 140, "unit": "l/s"},
        "Pmax": {"value": 22, "unit": "bar"},
        "Pmin": {"value": 4, "unit": "bar"},
        "BP": {"value": 4, "unit": "bar"},
        "Imax": {"value": 28000, "unit": "A"},
    }
    
    # Create WaterFlow object from the flow_params
    flow = from_flow_params(flow_params)
    
    print(f"\nWaterFlow object created:")
    print(f"  Pump speed range: {flow.pump_speed_min} - {flow.pump_speed_max} rpm")
    print(f"  Flow range: {flow.flow_min} - {flow.flow_max} l/s")
    print(f"  Pressure range: {flow.pressure_min} - {flow.pressure_max} bar")
    print(f"  Max current: {flow.current_max} A")
    
    # Use the WaterFlow object to compute operating parameters
    current = 20000  # A
    print(f"\nAt operating current {current} A:")
    print(f"  Pump speed: {flow.pump_speed(current):.2f} rpm")
    print(f"  Flow rate: {flow.flow_rate(current):.6f} m³/s")
    print(f"  Pressure: {flow.pressure(current):.2f} bar")
    print(f"  Pressure drop: {flow.pressure_drop(current):.2f} bar")
    print()


def example_from_database_record():
    """
    Example 2: Create WaterFlow from database record with custom field mapping
    
    This is useful when your database schema uses different field names.
    """
    print("=" * 70)
    print("Example 2: Creating WaterFlow from database record")
    print("=" * 70)
    
    # Simulated database record with custom field names
    db_record = {
        "min_pump_rpm": 1000,
        "max_pump_rpm": 2840,
        "min_flow_rate": 0,
        "max_flow_rate": 140,
        "max_pressure": 22,
        "min_pressure": 4,
        "back_pressure": 4,
        "max_current": 28000,
    }
    
    # Define mapping from database fields to flow parameter names
    key_mapping = {
        "Vp0": "min_pump_rpm",
        "Vpmax": "max_pump_rpm",
        "F0": "min_flow_rate",
        "Fmax": "max_flow_rate",
        "Pmax": "max_pressure",
        "Pmin": "min_pressure",
        "BP": "back_pressure",
        "Imax": "max_current",
    }
    
    # Create WaterFlow object with mapping
    flow = from_database_record(db_record, key_mapping)
    
    print(f"\nWaterFlow object created from database record")
    print(f"  Max flow rate: {flow.flow_max} l/s")
    print(f"  Max current: {flow.current_max} A")
    print()


def example_from_fitted_data():
    """
    Example 3: Create WaterFlow from fitted curve parameters
    
    This matches what the compute() function does when fitting experimental data.
    """
    print("=" * 70)
    print("Example 3: Creating WaterFlow from fitted curve parameters")
    print("=" * 70)
    
    # These parameters come from curve fitting:
    # Vp = Vpmax * (I/Imax)^2 + Vp0
    # F = F0 + Fmax * Vp/(Vpmax + Vp0)
    # P = Pmin + Pmax * (Vp/(Vpmax + Vp0))^2
    
    pump_speed_fit = (2840, 1000)  # (Vpmax, Vp0) from fitting
    flow_rate_fit = (0, 140)       # (F0, Fmax) from fitting
    pressure_fit = (4, 22)         # (Pmin, Pmax) from fitting
    back_pressure = 4.0
    max_current = 28000
    
    flow = from_fitted_data(
        pump_speed_fit,
        flow_rate_fit,
        pressure_fit,
        back_pressure,
        max_current
    )
    
    print(f"\nWaterFlow object created from fitted parameters")
    print(f"  Fitted pump speed: Vp = {flow.pump_speed_max}*(I/Imax)^2 + {flow.pump_speed_min}")
    print(f"  Fitted flow range: {flow.flow_min} - {flow.flow_max} l/s")
    print(f"  Fitted pressure range: {flow.pressure_min} - {flow.pressure_max} bar")
    print()


def example_using_in_compute():
    """
    Example 4: How to integrate into the compute() workflow
    
    Shows how to modify the compute() function to use waterflow_factory.
    """
    print("=" * 70)
    print("Example 4: Integration with compute() workflow")
    print("=" * 70)
    
    print("""
In examples/flow_params.py, the compute() function can be modified to use
the factory at the end:

    # At the end of compute() function, after flow_params is built:
    
    from python_magnetcooling.waterflow_factory import from_flow_params
    
    # Create WaterFlow object from the computed parameters
    waterflow = from_flow_params(flow_params)
    
    # Save both the raw parameters and use the WaterFlow object
    with open(filename, "w") as f:
        f.write(json.dumps(flow_params, indent=4))
    
    # Now you can use waterflow object for calculations
    velocity = waterflow.velocity(current=20000, cross_section=1e-4)
    
    return waterflow  # Return the object instead of just saving params
    """)
    print()


def example_load_from_saved_json():
    """
    Example 5: Load from saved JSON file (backward compatible)
    
    The existing WaterFlow.from_file() method still works, but you can also
    load the JSON and use the factory.
    """
    print("=" * 70)
    print("Example 5: Loading from saved JSON file")
    print("=" * 70)
    
    print("""
If you have a saved flow_params JSON file from compute(), you can load it
in two ways:

Method 1 - Using existing WaterFlow.from_file():
    from python_magnetcooling import WaterFlow
    flow = WaterFlow.from_file("M9_M10-flow_params.json")

Method 2 - Using the factory (more flexible):
    import json
    from python_magnetcooling.waterflow_factory import from_flow_params
    
    with open("M9_M10-flow_params.json", "r") as f:
        params = json.load(f)
    
    flow = from_flow_params(params)
    
Both methods produce the same result!
    """)
    print()


def main():
    """Run all examples"""
    print("\n")
    print("*" * 70)
    print("WaterFlow Factory Examples")
    print("Extracting WaterFlow object creation from database records")
    print("*" * 70)
    print("\n")
    
    example_from_flow_params()
    example_from_database_record()
    example_from_fitted_data()
    example_using_in_compute()
    example_load_from_saved_json()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
The waterflow_factory module provides clean separation between:
1. Data acquisition and fitting (examples/flow_params.py compute())
2. Object creation (waterflow_factory.from_flow_params())
3. Hydraulic calculations (WaterFlow methods)

This makes the code more maintainable and easier to test!
    """)


if __name__ == "__main__":
    main()
