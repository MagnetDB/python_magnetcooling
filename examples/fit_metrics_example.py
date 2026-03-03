"""
Example demonstrating fit quality metrics for hydraulic and hysteresis models.

This example shows how to:
1. Fit hydraulic system (pump, flow, pressure)
2. Compute fit quality metrics
3. Use metrics to decide if refit is needed
4. Fit hysteresis model and evaluate quality
"""

import numpy as np
from python_magnetcooling.fitting import (
    fit_hydraulic_system,
    fit_hysteresis_parameters,
    compute_pump_fit_metrics,
    compute_flow_fit_metrics,
    compute_pressure_fit_metrics,
    compute_all_hydraulic_metrics,
    compute_hysteresis_fit_metrics,
)
from python_magnetcooling.hysteresis import (
    compute_hysteresis_fit_metrics as compute_hyst_metrics_direct,
)


def example_hydraulic_metrics():
    """Example: Evaluate hydraulic fit quality"""
    print("\n" + "=" * 70)
    print("Example 1: Hydraulic System Fit Quality Metrics")
    print("=" * 70)

    # Generate synthetic data
    np.random.seed(42)
    current = np.linspace(0, 28000, 150)

    # Synthetic pump speed with quadratic relationship + noise
    pump_speed = 1000 + 1840 * (current / 28000) ** 2 + np.random.normal(0, 30, 150)

    # Synthetic flow rate
    flow_rate = 5 + 56 * (pump_speed - 1000) / 1840 + np.random.normal(0, 1.5, 150)

    # Synthetic pressure
    pressure = 4 + 18 * ((pump_speed - 1000) / 1840) ** 2 + np.random.normal(
        0, 0.3, 150
    )

    # Back pressure
    back_pressure = 4 * np.ones(150) + np.random.normal(0, 0.1, 150)

    # Fit the system
    print("\nFitting hydraulic system...")
    pump_fit, flow_pressure_fit = fit_hydraulic_system(
        current=current,
        pump_speed=pump_speed,
        flow_rate=flow_rate,
        pressure=pressure,
        back_pressure=back_pressure,
        imax=28000,
        method="simple",
    )

    print(
        f"  Pump: Vp0={pump_fit.vp0:.1f} rpm, Vpmax={pump_fit.vpmax:.1f} rpm, Imax={pump_fit.imax} A"
    )
    print(
        f"  Flow: F0={flow_pressure_fit.f0:.2f} l/s, Fmax={flow_pressure_fit.fmax:.2f} l/s"
    )
    print(
        f"  Pressure: Pmin={flow_pressure_fit.pmin:.2f} bar, Pmax={flow_pressure_fit.pmax:.2f} bar"
    )

    # Compute metrics individually
    print("\n--- Individual Metrics ---")

    pump_metrics = compute_pump_fit_metrics(current, pump_speed, pump_fit)
    print(f"\nPump Fit Metrics:")
    print(f"  RMSE:      {pump_metrics.rmse:.3f} rpm")
    print(f"  MAE:       {pump_metrics.mae:.3f} rpm")
    print(f"  Max Error: {pump_metrics.max_error:.3f} rpm")
    print(f"  R²:        {pump_metrics.r_squared:.6f}")
    if pump_metrics.mape is not None:
        print(f"  MAPE:      {pump_metrics.mape:.2f}%")
    print(f"  Verdict:   {'✓ Good fit' if pump_metrics.is_good_fit() else '✗ Refit needed'}")

    flow_metrics = compute_flow_fit_metrics(
        current, flow_rate, pump_fit, flow_pressure_fit
    )
    print(f"\nFlow Fit Metrics:")
    print(f"  RMSE:      {flow_metrics.rmse:.3f} l/s")
    print(f"  MAE:       {flow_metrics.mae:.3f} l/s")
    print(f"  Max Error: {flow_metrics.max_error:.3f} l/s")
    print(f"  R²:        {flow_metrics.r_squared:.6f}")
    if flow_metrics.mape is not None:
        print(f"  MAPE:      {flow_metrics.mape:.2f}%")
    print(f"  Verdict:   {'✓ Good fit' if flow_metrics.is_good_fit() else '✗ Refit needed'}")

    pressure_metrics = compute_pressure_fit_metrics(
        current, pressure, pump_fit, flow_pressure_fit
    )
    print(f"\nPressure Fit Metrics:")
    print(f"  RMSE:      {pressure_metrics.rmse:.3f} bar")
    print(f"  MAE:       {pressure_metrics.mae:.3f} bar")
    print(f"  Max Error: {pressure_metrics.max_error:.3f} bar")
    print(f"  R²:        {pressure_metrics.r_squared:.6f}")
    if pressure_metrics.mape is not None:
        print(f"  MAPE:      {pressure_metrics.mape:.2f}%")
    print(
        f"  Verdict:   {'✓ Good fit' if pressure_metrics.is_good_fit() else '✗ Refit needed'}"
    )

    # Compute all metrics at once
    print("\n--- All Metrics Summary ---")
    all_metrics = compute_all_hydraulic_metrics(
        current, pump_speed, flow_rate, pressure, pump_fit, flow_pressure_fit
    )

    print("\nComprehensive Quality Report:")
    for component, metrics in all_metrics.items():
        status = "✓ GOOD" if metrics.is_good_fit(r_squared_threshold=0.90) else "✗ POOR"
        print(
            f"  {component.upper():8s}: RMSE={metrics.rmse:6.3f}, R²={metrics.r_squared:.4f}  [{status}]"
        )

    # Decision logic
    print("\n--- Refit Recommendations ---")
    poor_fits = [
        name
        for name, m in all_metrics.items()
        if not m.is_good_fit(r_squared_threshold=0.90)
    ]

    if poor_fits:
        print(f"⚠ Consider refitting: {', '.join(poor_fits)}")
        print("  Possible causes:")
        print("    - Insufficient data coverage")
        print("    - Outliers affecting fit")
        print("    - Model assumptions violated")
        print("    - Operating conditions changed")
    else:
        print("✓ All fits are acceptable. No refit needed.")


def example_hysteresis_metrics():
    """Example: Evaluate hysteresis fit quality"""
    print("\n" + "=" * 70)
    print("Example 2: Hysteresis Model Fit Quality Metrics")
    print("=" * 70)

    np.random.seed(42)

    # Generate synthetic hysteresis data
    x_up = np.linspace(0, 20, 100)
    x_down = np.linspace(20, 0, 100)
    x = np.concatenate([x_up, x_down])

    # Create output with two-level hysteresis + noise
    y_up = np.where(x_up > 8, 200, 100) + np.random.normal(0, 4, 100)
    y_down = np.where(x_down < 6, 100, 200) + np.random.normal(0, 4, 100)
    y = np.concatenate([y_up, y_down])

    # Fit hysteresis parameters
    print("\nFitting hysteresis model...")
    hyst_fit = fit_hysteresis_parameters(
        power=x, flow_rate=y, n_levels=None, verbose=False
    )

    print(f"  Found {len(hyst_fit.thresholds)} hysteresis levels")
    for i, (asc, desc) in enumerate(hyst_fit.thresholds):
        print(
            f"    Level {i}: asc={asc:.2f}, desc={desc:.2f}, "
            f"low={hyst_fit.low_values[i]:.1f}, high={hyst_fit.high_values[i]:.1f}"
        )

    # Compute metrics using fitting module
    print("\n--- Fit Quality Metrics (via fitting module) ---")
    metrics = compute_hysteresis_fit_metrics(x, y, hyst_fit)
    print(metrics)

    # Compute metrics using hysteresis module directly
    print("\n--- Fit Quality Metrics (via hysteresis module) ---")
    metrics_dict = compute_hyst_metrics_direct(
        x, y, hyst_fit.thresholds, hyst_fit.low_values, hyst_fit.high_values
    )

    for key, value in metrics_dict.items():
        if key == "mape":
            print(f"  {key.upper():12s}: {value:.2f}%")
        elif key == "n_points":
            print(f"  {key.upper():12s}: {value}")
        elif key == "match_rate":
            print(f"  {key.upper():12s}: {value:.1f}%")
        else:
            print(f"  {key.upper():12s}: {value:.4f}")

    # Decision logic
    print("\n--- Evaluation ---")
    r_squared = metrics_dict["r_squared"]
    rmse = metrics_dict["rmse"]
    match_rate = metrics_dict["match_rate"]

    print(f"R² = {r_squared:.4f}")
    if r_squared > 0.95:
        print("  ✓ Excellent fit quality")
    elif r_squared > 0.85:
        print("  ✓ Good fit quality")
    elif r_squared > 0.70:
        print("  ⚠ Fair fit - consider refinement")
    else:
        print("  ✗ Poor fit - refit recommended")

    y_range = y.max() - y.min()
    rmse_percent = (rmse / y_range) * 100
    print(f"\nRMSE = {rmse:.2f} ({rmse_percent:.1f}% of range)")
    if rmse_percent < 5:
        print("  ✓ Excellent accuracy")
    elif rmse_percent < 10:
        print("  ✓ Good accuracy")
    elif rmse_percent < 15:
        print("  ⚠ Moderate accuracy")
    else:
        print("  ✗ Poor accuracy - refit recommended")

    print(f"\nMatch Rate = {match_rate:.1f}%")
    if match_rate > 90:
        print("  ✓ Excellent prediction rate")
    elif match_rate > 80:
        print("  ✓ Good prediction rate")
    else:
        print("  ⚠ Low prediction rate - check data quality")

    # Recommendations
    print("\n--- Recommendations ---")
    if r_squared < 0.85 or rmse_percent > 10:
        print("⚠ Refit suggested. Consider:")
        print("  1. Try different n_levels parameter")
        print("  2. Remove outliers before fitting")
        print("  3. Check if data exhibits true hysteresis")
        print("  4. Verify data quality and coverage")
    else:
        print("✓ Fit quality is acceptable. No immediate action needed.")


def example_refit_decision():
    """Example: Automated refit decision logic"""
    print("\n" + "=" * 70)
    print("Example 3: Automated Refit Decision Logic")
    print("=" * 70)

    # Simulate checking multiple datasets
    print("\nSimulating quality checks on multiple historical fits...\n")

    # Dataset 1: Good fit
    np.random.seed(1)
    current_1 = np.linspace(0, 28000, 100)
    pump_speed_1 = 1000 + 1840 * (current_1 / 28000) ** 2 + np.random.normal(0, 10, 100)

    # Dataset 2: Poor fit (high noise)
    np.random.seed(2)
    current_2 = np.linspace(0, 28000, 100)
    pump_speed_2 = 1000 + 1840 * (current_2 / 28000) ** 2 + np.random.normal(0, 150, 100)

    datasets = [
        ("Dataset 2024-01-15", current_1, pump_speed_1),
        ("Dataset 2024-03-03", current_2, pump_speed_2),
    ]

    for name, curr, pspeed in datasets:
        # Fit
        from python_magnetcooling.fitting import fit_pump_speed

        pump_fit = fit_pump_speed(curr, pspeed, imax=28000, method="simple")

        # Compute metrics
        metrics = compute_pump_fit_metrics(curr, pspeed, pump_fit)

        # Decision
        print(f"{name}:")
        print(f"  RMSE = {metrics.rmse:.2f} rpm, R² = {metrics.r_squared:.4f}")

        # Automated decision with thresholds
        needs_refit = False
        reasons = []

        if metrics.r_squared < 0.90:
            needs_refit = True
            reasons.append(f"Low R² ({metrics.r_squared:.4f} < 0.90)")

        if metrics.rmse > 50:
            needs_refit = True
            reasons.append(f"High RMSE ({metrics.rmse:.1f} > 50 rpm)")

        if metrics.max_error > 200:
            needs_refit = True
            reasons.append(f"High max error ({metrics.max_error:.1f} > 200 rpm)")

        if needs_refit:
            print(f"  ✗ REFIT RECOMMENDED")
            for reason in reasons:
                print(f"    - {reason}")
        else:
            print(f"  ✓ FIT ACCEPTABLE")

        print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Fit Quality Metrics Examples for python_magnetcooling")
    print("=" * 70)

    example_hydraulic_metrics()
    example_hysteresis_metrics()
    example_refit_decision()

    print("\n" + "=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Use compute_*_fit_metrics() to evaluate fit quality")
    print("  2. Check R² > 0.90 and RMSE < acceptable threshold")
    print("  3. Use is_good_fit() for quick quality checks")
    print("  4. Refit if metrics indicate poor quality")
    print("  5. Consider outlier removal and data quality issues")
    print()
