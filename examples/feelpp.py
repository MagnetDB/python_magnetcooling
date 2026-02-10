"""
Example: Using thermohydraulics.py independently
"""

from python_magnetcooling.thermohydraulics import (
    ThermalHydraulicCalculator,
    ThermalHydraulicInput,
    ChannelInput,
    ChannelGeometry,
    compute_single_channel,
)
from python_magnetcooling.waterflow import waterflow

# Example 1: Single channel quick calculation
print("=== Single Channel Example ===")
result = compute_single_channel(
    hydraulic_diameter=0.008,  # 8 mm
    cross_section=5e-5,  # 50 mm²
    length=0.5,  # 50 cm
    power=50000,  # 50 kW
    temp_inlet=290.0,  # K
    pressure_inlet=15.0,  # bar
    pressure_drop=5.0,  # bar
    verbose=True,
)

print("\nResults:")
print(f"  Outlet temperature: {result.temp_outlet:.2f} K")
print(f"  Temperature rise: {result.temp_rise:.2f} K")
print(f"  Heat coefficient: {result.heat_coeff:.1f} W/m²/K")
print(f"  Velocity: {result.velocity:.2f} m/s")
print(f"  Flow rate: {result.flow_rate*1000:.2f} l/s")

# Example 2: Multiple channels with waterflow
print("\n=== Multiple Channels with Waterflow ===")

# Load pump characteristics
flow_params = waterflow.flow_params("magnet-flow_params.json")

# Define channels
channels = [
    ChannelInput(
        geometry=ChannelGeometry(
            hydraulic_diameter=0.008, cross_section=5e-5, length=0.5, name="Channel_1"
        ),
        power=40000,
        temp_inlet=290.0,
    ),
    ChannelInput(
        geometry=ChannelGeometry(
            hydraulic_diameter=0.010, cross_section=7e-5, length=0.5, name="Channel_2"
        ),
        power=60000,
        temp_inlet=290.0,
    ),
]

# Create inputs (pressure will be overridden by waterflow)
inputs = ThermalHydraulicInput(
    channels=channels,
    pressure_inlet=15.0,  # Will be updated
    pressure_drop=5.0,  # Will be updated
    heat_correlation="Montgomery",
    friction_model="Constant",
)

# Compute with waterflow
calc = ThermalHydraulicCalculator(verbose=True)
result = calc.compute_from_waterflow(inputs, flow_params, current=30000)

print("\nGlobal Results:")
print(f"  Total flow: {result.total_flow_rate*1000:.2f} l/s")
print(f"  Mixed outlet temp: {result.outlet_temp_mixed:.2f} K")
print(f"  Total power: {result.total_power/1000:.1f} kW")

for i, ch in enumerate(result.channels):
    print(f"\nChannel {i+1}:")
    print(f"  Outlet temp: {ch.temp_outlet:.2f} K")
    print(f"  Velocity: {ch.velocity:.2f} m/s")
    print(f"  Heat coeff: {ch.heat_coeff:.1f} W/m²/K")
