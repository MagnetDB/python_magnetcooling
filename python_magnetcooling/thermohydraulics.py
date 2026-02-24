"""
Thermal-hydraulic calculations for cooling channels

This module is independent of FeelPP and can be used for any
water-cooled magnet thermal analysis.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .cooling import steam, Uw, getDT, getHeatCoeff, getTout
from .waterflow import WaterFlow


@dataclass
class ChannelGeometry:
    """Geometric parameters for a cooling channel"""

    hydraulic_diameter: float  # m
    cross_section: float  # m²
    length: float  # m
    name: str = ""

    def __post_init__(self):
        if self.hydraulic_diameter <= 0:
            raise ValueError("Hydraulic diameter must be positive")
        if self.cross_section <= 0:
            raise ValueError("Cross section must be positive")
        if self.length <= 0:
            raise ValueError("Length must be positive")


@dataclass
class AxialDiscretization:
    """Axial discretization for gradHZ mode"""

    z_positions: List[float]  # m, axial positions
    power_distribution: List[float]  # W, power per section

    def __post_init__(self):
        if len(self.z_positions) != len(self.power_distribution) + 1:
            raise ValueError(
                f"z_positions length ({len(self.z_positions)}) must be "
                f"power_distribution length + 1 ({len(self.power_distribution) + 1})"
            )


@dataclass
class ChannelInput:
    """Input parameters for a single cooling channel"""

    geometry: ChannelGeometry
    power: float  # W, total power dissipated
    temp_inlet: float  # K, inlet water temperature
    axial_discretization: Optional[AxialDiscretization] = None

    # Initial guesses (optional, for faster convergence)
    temp_outlet_guess: Optional[float] = None
    heat_coeff_guess: Optional[float] = None
    velocity_guess: Optional[float] = None


@dataclass
class ChannelOutput:
    """Results for a single cooling channel"""

    # Flow parameters
    velocity: float  # m/s
    flow_rate: float  # m³/s
    friction_factor: float  # dimensionless

    # Thermal parameters
    temp_inlet: float  # K
    temp_outlet: float  # K
    temp_rise: float  # K (outlet - inlet)
    temp_mean: float  # K (average)

    # Heat transfer
    heat_coeff: float  # W/m²/K
    heat_coeff_distribution: Optional[List[float]] = None  # W/m²/K

    # Temperature distribution (for gradHZ mode)
    temp_distribution: Optional[List[float]] = None  # K

    # Water properties at outlet
    density_outlet: float = 0.0  # kg/m³
    specific_heat_outlet: float = 0.0  # J/kg/K

    # Convergence info
    converged: bool = True
    iterations: int = 0


@dataclass
class ThermalHydraulicInput:
    """Complete input for thermal-hydraulic analysis"""

    channels: List[ChannelInput]
    pressure_inlet: float  # bar
    pressure_drop: float  # bar

    # Heat correlation parameters
    heat_correlation: str = "Montgomery"  # Montgomery, Dittus, Colburn, Silverberg
    friction_model: str = "Constant"  # Constant, Blasius, Filonenko, Colebrook, Swamee
    fuzzy_factor: float = 1.0  # Heat correlation correction factor
    extra_pressure_loss: float = 1.0  # Additional pressure loss coefficient

    # Convergence parameters
    max_iterations: int = 10
    tolerance_flow: float = 1e-3
    tolerance_temp: float = 1e-3
    relaxation_factor: float = 0.0  # 0 = no relaxation, 1 = full old value


@dataclass
class ThermalHydraulicOutput:
    """Complete thermal-hydraulic analysis results"""

    channels: List[ChannelOutput]

    # Global results
    total_flow_rate: float  # m³/s, sum of all channels
    outlet_temp_mixed: float  # K, mixed outlet temperature
    total_power: float  # W, sum of all powers

    # Convergence info
    max_error_temp: float = 0.0  # Maximum temperature error
    max_error_heat_coeff: float = 0.0  # Maximum heat coefficient error
    converged: bool = True


class ThermalHydraulicCalculator:
    """
    Standalone thermal-hydraulic calculator for water-cooled magnets

    Can be used independently of FeelPP for:
    - Design calculations
    - Sensitivity studies
    - Optimization
    - Validation against experimental data

    Example:
        >>> calc = ThermalHydraulicCalculator()
        >>>
        >>> # Define channel geometry
        >>> channel = ChannelInput(
        ...     geometry=ChannelGeometry(
        ...         hydraulic_diameter=0.008,
        ...         cross_section=5e-5,
        ...         length=0.5,
        ...         name="Channel_1"
        ...     ),
        ...     power=50000,
        ...     temp_inlet=290.0
        ... )
        >>>
        >>> # Run calculation
        >>> inputs = ThermalHydraulicInput(
        ...     channels=[channel],
        ...     pressure_inlet=15.0,
        ...     pressure_drop=5.0
        ... )
        >>> result = calc.compute(inputs)
        >>> print(f"Outlet temp: {result.channels[0].temp_outlet:.2f} K")
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize calculator

        Args:
            verbose: If True, print detailed convergence information
        """
        self.verbose = verbose

    def compute(self, inputs: ThermalHydraulicInput) -> ThermalHydraulicOutput:
        """
        Compute thermal-hydraulic solution for all channels

        Args:
            inputs: Complete input specification

        Returns:
            Complete thermal-hydraulic solution

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If solution does not converge
        """
        self._validate_inputs(inputs)

        channel_outputs = []
        errors_temp = []
        errors_heat = []

        for i, channel in enumerate(inputs.channels):
            if self.verbose:
                print(f"\n=== Channel {i}: {channel.geometry.name} ===")

            if channel.axial_discretization:
                # gradHZ mode with axial discretization
                output = self._compute_channel_axial(channel, inputs)
            else:
                # Standard mode (mean values)
                output = self._compute_channel_uniform(channel, inputs)

            channel_outputs.append(output)

            # Compute errors if we have initial guesses
            if channel.temp_outlet_guess:
                err_t = abs(1 - channel.temp_outlet_guess / output.temp_outlet)
                errors_temp.append(err_t)

            if channel.heat_coeff_guess:
                err_h = abs(1 - channel.heat_coeff_guess / output.heat_coeff)
                errors_heat.append(err_h)

        # Compute mixed outlet temperature
        outlet_temp_mixed = self._compute_mixed_outlet_temp(channel_outputs, inputs.pressure_inlet)

        total_flow = sum(ch.flow_rate for ch in channel_outputs)
        total_power = sum(ch_in.power for ch_in in inputs.channels)

        return ThermalHydraulicOutput(
            channels=channel_outputs,
            total_flow_rate=total_flow,
            outlet_temp_mixed=outlet_temp_mixed,
            total_power=total_power,
            max_error_temp=max(errors_temp) if errors_temp else 0.0,
            max_error_heat_coeff=max(errors_heat) if errors_heat else 0.0,
            converged=all(ch.converged for ch in channel_outputs),
        )

    def compute_from_waterflow(
        self, inputs: ThermalHydraulicInput, waterflow_params: WaterFlow, current: float
    ) -> ThermalHydraulicOutput:
        """
        Compute thermal-hydraulics using waterflow pump characteristics

        Args:
            inputs: Thermal-hydraulic inputs (pressure values will be overridden)
            waterflow_params: Pump characteristic curves
            current: Operating current [A]

        Returns:
            Complete thermal-hydraulic solution with flow-adjusted pressures
        """
        # Override pressure values from waterflow
        inputs.pressure_inlet = waterflow_params.pressure(current)
        inputs.pressure_drop = waterflow_params.pressure_drop(current)

        # Compute total cross-section
        total_section = sum(ch.geometry.cross_section for ch in inputs.channels)

        # Get mean velocity from waterflow
        u_mean_from_flow = waterflow_params.velocity(current, total_section)

        # Set velocity guesses for faster convergence
        for channel in inputs.channels:
            if channel.velocity_guess is None:
                channel.velocity_guess = u_mean_from_flow

        return self.compute(inputs)

    def _compute_channel_uniform(
        self, channel: ChannelInput, global_inputs: ThermalHydraulicInput
    ) -> ChannelOutput:
        """Compute channel with uniform (mean) properties"""

        geom = channel.geometry

        # Initial guesses
        U = channel.velocity_guess if channel.velocity_guess is not None else 5.0  # m/s
        dT = (
            channel.temp_outlet_guess - channel.temp_inlet
            if channel.temp_outlet_guess is not None
            else 10.0
        )
        h = channel.heat_coeff_guess if channel.heat_coeff_guess is not None else 80000.0
        cf = 0.055

        converged = False
        iteration = 0

        for iteration in range(global_inputs.max_iterations):
            # Current estimates
            T_mean = channel.temp_inlet + dT / 2.0
            flow = U * geom.cross_section

            # Compute new temperature rise
            dT_new = getDT(flow, channel.power, T_mean, global_inputs.pressure_inlet)

            # Compute new heat transfer coefficient
            h_new = getHeatCoeff(
                geom.hydraulic_diameter,
                geom.length,
                U,
                T_mean,
                global_inputs.pressure_inlet,
                global_inputs.pressure_drop,
                model=global_inputs.heat_correlation,
                friction=global_inputs.friction_model,
                fuzzy=global_inputs.fuzzy_factor,
                pextra=global_inputs.extra_pressure_loss,
            )

            # Compute new velocity from pressure drop
            Steam = steam(T_mean, global_inputs.pressure_inlet)
            U_new, cf_new = Uw(
                Steam,
                global_inputs.pressure_drop,
                geom.hydraulic_diameter,
                geom.length,
                friction=global_inputs.friction_model,
                uguess=U,
            )

            # Check convergence
            flow_new = U_new * geom.cross_section
            err_flow = abs(1 - flow_new / flow) if flow > 0 else 1.0
            err_temp = abs(1 - dT_new / dT) if dT > 0 else 1.0

            if self.verbose:
                print(
                    f"  Iter {iteration}: U={U:.3f}, dT={dT:.3f}, "
                    f"h={h:.1f}, err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
                )

            # Update with relaxation
            relax = global_inputs.relaxation_factor
            U = (1 - relax) * U_new + relax * U
            dT = (1 - relax) * dT_new + relax * dT
            h = (1 - relax) * h_new + relax * h
            cf = cf_new

            if err_flow < global_inputs.tolerance_flow and err_temp < global_inputs.tolerance_temp:
                converged = True
                break

        # Compute outlet properties
        T_outlet = channel.temp_inlet + dT
        Steam_outlet = steam(T_outlet, global_inputs.pressure_inlet)

        return ChannelOutput(
            velocity=U,
            flow_rate=U * geom.cross_section,
            friction_factor=cf,
            temp_inlet=channel.temp_inlet,
            temp_outlet=T_outlet,
            temp_rise=dT,
            temp_mean=channel.temp_inlet + dT / 2.0,
            heat_coeff=h,
            density_outlet=Steam_outlet.rho,
            specific_heat_outlet=Steam_outlet.cp * Steam_outlet.rho,
            converged=converged,
            iterations=iteration + 1,
        )

    def _compute_channel_axial(
        self, channel: ChannelInput, global_inputs: ThermalHydraulicInput
    ) -> ChannelOutput:
        """Compute channel with axial discretization (gradHZ mode)"""

        geom = channel.geometry
        axial = channel.axial_discretization
        n_sections = len(axial.power_distribution)

        # Initialize distributions
        T_z = [channel.temp_inlet] + [channel.temp_inlet + 10.0] * n_sections
        h_z = [80000.0] * (n_sections + 1)

        # Initial velocity guess
        U = channel.velocity_guess if channel.velocity_guess is not None else 5.0

        converged = False
        iteration = 0

        for iteration in range(global_inputs.max_iterations):
            flow = U * geom.cross_section
            T_z_old = T_z.copy()

            # Compute temperature distribution
            for k in range(n_sections):
                # Local pressure
                z_frac = (axial.z_positions[k] - axial.z_positions[0]) / (
                    axial.z_positions[-1] - axial.z_positions[0]
                )
                P_local = global_inputs.pressure_inlet - global_inputs.pressure_drop * z_frac

                # Temperature rise in this section
                T_mean_section = (T_z[k] + T_z[k + 1]) / 2.0
                dT_section = getDT(flow, axial.power_distribution[k], T_mean_section, P_local)
                T_z[k + 1] = T_z[k] + dT_section

            # Compute heat coefficient distribution
            for k in range(len(T_z)):
                z_frac = (axial.z_positions[k] - axial.z_positions[0]) / (
                    axial.z_positions[-1] - axial.z_positions[0]
                )
                P_local = global_inputs.pressure_inlet - global_inputs.pressure_drop * z_frac

                h_z[k] = getHeatCoeff(
                    geom.hydraulic_diameter,
                    geom.length,
                    U,
                    T_z[k],
                    P_local,
                    global_inputs.pressure_drop,
                    model=global_inputs.heat_correlation,
                    friction=global_inputs.friction_model,
                    fuzzy=global_inputs.fuzzy_factor,
                    pextra=global_inputs.extra_pressure_loss,
                )

            # Update velocity
            T_mean = (T_z[0] + T_z[-1]) / 2.0
            Steam = steam(T_mean, global_inputs.pressure_inlet)
            U_new, cf = Uw(
                Steam,
                global_inputs.pressure_drop,
                geom.hydraulic_diameter,
                geom.length,
                friction=global_inputs.friction_model,
                uguess=U,
            )

            # Check convergence
            flow_new = U_new * geom.cross_section
            err_flow = abs(1 - flow_new / (U * geom.cross_section))
            err_temp = max(abs(1 - T_new / T_old) for T_new, T_old in zip(T_z[1:], T_z_old[1:]))

            if self.verbose:
                print(
                    f"  Iter {iteration}: U={U:.3f}, T_out={T_z[-1]:.3f}, "
                    f"err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
                )

            U = U_new

            if err_flow < global_inputs.tolerance_flow and err_temp < global_inputs.tolerance_temp:
                converged = True
                break

        # Compute outlet properties
        Steam_outlet = steam(T_z[-1], global_inputs.pressure_inlet)

        return ChannelOutput(
            velocity=U,
            flow_rate=U * geom.cross_section,
            friction_factor=cf,
            temp_inlet=T_z[0],
            temp_outlet=T_z[-1],
            temp_rise=T_z[-1] - T_z[0],
            temp_mean=np.mean(T_z),
            heat_coeff=np.mean(h_z),
            heat_coeff_distribution=h_z,
            temp_distribution=T_z,
            density_outlet=Steam_outlet.rho,
            specific_heat_outlet=Steam_outlet.cp * Steam_outlet.rho,
            converged=converged,
            iterations=iteration + 1,
        )

    def _compute_mixed_outlet_temp(self, channels: List[ChannelOutput], pressure: float) -> float:
        """Compute mixed outlet temperature from all channels"""

        T_list = [ch.temp_outlet for ch in channels]
        rho_list = [ch.density_outlet for ch in channels]
        cp_list = [ch.specific_heat_outlet for ch in channels]
        Q_list = [ch.flow_rate for ch in channels]

        return getTout(T_list, rho_list, cp_list, Q_list)

    def _validate_inputs(self, inputs: ThermalHydraulicInput):
        """Validate input parameters"""
        if not inputs.channels:
            raise ValueError("At least one channel required")

        if inputs.pressure_inlet <= 0:
            raise ValueError("Inlet pressure must be positive")

        if inputs.pressure_drop < 0:
            raise ValueError("Pressure drop cannot be negative")

        for i, channel in enumerate(inputs.channels):
            if channel.power < 0:
                raise ValueError(f"Channel {i}: Power cannot be negative")

            if channel.temp_inlet <= 0:
                raise ValueError(f"Channel {i}: Inlet temperature must be positive")


# Convenience functions for common use cases


def compute_single_channel(
    hydraulic_diameter: float,
    cross_section: float,
    length: float,
    power: float,
    temp_inlet: float,
    pressure_inlet: float,
    pressure_drop: float,
    heat_correlation: str = "Montgomery",
    friction_model: str = "Constant",
    verbose: bool = False,
) -> ChannelOutput:
    """
    Quick calculation for a single channel

    Args:
        hydraulic_diameter: Hydraulic diameter [m]
        cross_section: Cross-sectional area [m²]
        length: Channel length [m]
        power: Power dissipated [W]
        temp_inlet: Inlet temperature [K]
        pressure_inlet: Inlet pressure [bar]
        pressure_drop: Pressure drop [bar]
        heat_correlation: Heat transfer correlation
        friction_model: Friction factor model
        verbose: Print convergence info

    Returns:
        Channel thermal-hydraulic solution
    """
    calc = ThermalHydraulicCalculator(verbose=verbose)

    channel = ChannelInput(
        geometry=ChannelGeometry(
            hydraulic_diameter=hydraulic_diameter, cross_section=cross_section, length=length
        ),
        power=power,
        temp_inlet=temp_inlet,
    )

    inputs = ThermalHydraulicInput(
        channels=[channel],
        pressure_inlet=pressure_inlet,
        pressure_drop=pressure_drop,
        heat_correlation=heat_correlation,
        friction_model=friction_model,
    )

    result = calc.compute(inputs)
    return result.channels[0]
