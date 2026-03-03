"""
Example demonstrating plotting functions for hydraulic fits and hysteresis models.

This example shows how to use the plotting utilities to visualize:
- Pump speed fits
- Flow and pressure fits
- Hysteresis models and fits
"""

import numpy as np
import pandas as pd
from python_magnetcooling.fitting import (
    fit_hydraulic_system,
    fit_hysteresis_parameters,
    plot_pump_fit,
    plot_flow_pressure_fit,
    plot_hysteresis_fit,
)
from python_magnetcooling.hysteresis import (
    plot_hysteresis_model,
    plot_hysteresis_fit as plot_hyst_direct,
)


def example_pump_flow_pressure_plotting():
    """Example: Plot pump, flow, and pressure fits from synthetic data"""
    print("\n=== Example 1: Pump/Flow/Pressure Fitting ===")
    
    # Generate synthetic data
    np.random.seed(42)
    current = np.linspace(0, 28000, 100)
    
    # Synthetic pump speed with quadratic relationship
    pump_speed = 1000 + 1840 * (current / 28000) ** 2 + np.random.normal(0, 50, 100)
    
    # Synthetic flow rate (linear with pump speed)
    flow_rate = 5 + 56 * (pump_speed - 1000) / 1840 + np.random.normal(0, 2, 100)
    
    # Synthetic pressure (quadratic with pump speed)
    pressure = 4 + 18 * ((pump_speed - 1000) / 1840) ** 2 + np.random.normal(0, 0.5, 100)
    
    # Back pressure (constant with noise)
    back_pressure = 4 * np.ones(100) + np.random.normal(0, 0.2, 100)
    
    # Fit the system
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        current=current,
        pump_speed=pump_speed,
        flow_rate=flow_rate,
        pressure=pressure,
        back_pressure=back_pressure,
        imax=28000,
        method="simple",
    )
    
    print(f"Pump fit: Vp0={pump_fit.vp0:.1f}, Vpmax={pump_fit.vpmax:.1f}, Imax={pump_fit.imax}")
    print(f"Flow fit: F0={flow_pressure_fit.f0:.2f}, Fmax={flow_pressure_fit.fmax:.2f}")
    print(f"Pressure fit: Pmin={flow_pressure_fit.pmin:.2f}, Pmax={flow_pressure_fit.pmax:.2f}")
    
    # Plot pump fit
    plot_pump_fit(
        current,
        pump_speed,
        pump_fit,
        title="Pump Speed Fit Example",
        show=False,
        save_path="pump_fit_example.png",
    )
    print("✓ Saved pump_fit_example.png")
    
    # Plot flow and pressure fits
    plot_flow_pressure_fit(
        current,
        flow_rate,
        pressure,
        pump_fit,
        flow_pressure_fit,
        title="Flow and Pressure Fits Example",
        show=False,
        save_path="flow_pressure_fit_example.png",
    )
    print("✓ Saved flow_pressure_fit_example.png")


def example_hysteresis_model_plotting():
    """Example: Plot a hysteresis model with synthetic input signal"""
    print("\n=== Example 2: Hysteresis Model Visualization ===")
    
    # Create synthetic input signal (triangle wave)
    x_up = np.linspace(0, 15, 100)
    x_down = np.linspace(15, 0, 100)
    x = np.concatenate([x_up, x_down, x_up, x_down])
    
    # Define three-level hysteresis
    thresholds = [(3.0, 2.0), (8.0, 6.0), (12.0, 10.0)]
    low_values = [100, 200, 300]
    high_values = [250, 350, 450]
    
    print(f"Thresholds: {thresholds}")
    print(f"Low values: {low_values}")
    print(f"High values: {high_values}")
    
    # Plot the model
    plot_hysteresis_model(
        x,
        thresholds,
        low_values,
        high_values,
        xlabel="Power (MW)",
        ylabel="Flow Rate (m³/h)",
        title="Three-Level Hysteresis Model",
        show=False,
        save_path="hysteresis_model_example.png",
    )
    print("✓ Saved hysteresis_model_example.png")


def example_hysteresis_fit_plotting():
    """Example: Fit hysteresis to synthetic data and plot results"""
    print("\n=== Example 3: Hysteresis Fit Visualization ===")
    
    np.random.seed(42)
    
    # Generate synthetic data with hysteresis
    x_up = np.linspace(0, 20, 100)
    x_down = np.linspace(20, 0, 100)
    x = np.concatenate([x_up, x_down])
    
    # Create output with two-level hysteresis + noise
    y_up = np.where(x_up > 8, 200, 100) + np.random.normal(0, 3, 100)
    y_down = np.where(x_down < 6, 100, 200) + np.random.normal(0, 3, 100)
    y = np.concatenate([y_up, y_down])
    
    # Create DataFrame
    df = pd.DataFrame({"power": x, "flow": y})
    
    # Fit hysteresis parameters
    result = fit_hysteresis_parameters(
        power=x,
        flow_rate=y,
        n_levels=None,  # Auto-detect levels
        verbose=True,
    )
    
    print(f"\nFitted {len(result.thresholds)} hysteresis levels:")
    for i, (asc, desc) in enumerate(result.thresholds):
        print(f"  Level {i}: asc={asc:.2f}, desc={desc:.2f}, "
              f"low={result.low_values[i]:.1f}, high={result.high_values[i]:.1f}")
    
    # Plot the fit (method 1: using fitting.plot_hysteresis_fit)
    plot_hysteresis_fit(
        x,
        y,
        result,
        xlabel="Magnet Power (MW)",
        ylabel="Cooling Flow (m³/h)",
        title="Hysteresis Fit Example (via fitting module)",
        show=False,
        save_path="hysteresis_fit_example.png",
    )
    print("✓ Saved hysteresis_fit_example.png")
    
    # Plot the fit (method 2: using hysteresis.plot_hysteresis_fit directly)
    plot_hyst_direct(
        df,
        result.thresholds,
        result.low_values,
        result.high_values,
        x_col="power",
        y_col="flow",
        xlabel="Magnet Power (MW)",
        ylabel="Cooling Flow (m³/h)",
        title="Hysteresis Fit Example (via hysteresis module)",
        show=False,
        save_path="hysteresis_fit_direct_example.png",
    )
    print("✓ Saved hysteresis_fit_direct_example.png")


if __name__ == "__main__":
    print("Plotting Examples for python_magnetcooling")
    print("=" * 50)
    
    try:
        import matplotlib
        matplotlib_available = True
    except ImportError:
        matplotlib_available = False
        print("\n⚠ WARNING: matplotlib is not installed.")
        print("Install it with: pip install matplotlib")
        print("Plotting examples will not work without it.\n")
    
    if matplotlib_available:
        example_pump_flow_pressure_plotting()
        example_hysteresis_model_plotting()
        example_hysteresis_fit_plotting()
        
        print("\n" + "=" * 50)
        print("✓ All plotting examples completed successfully!")
        print("Check the current directory for generated PNG files.")
    else:
        print("Skipping examples due to missing matplotlib.")
