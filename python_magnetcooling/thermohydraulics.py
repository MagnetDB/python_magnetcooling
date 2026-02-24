"""
Thermal-hydraulic calculations for cooling channels

This module is independent of FeelPP and can be used for any
water-cooled magnet thermal analysis.
"""

from dataclasses import dataclass
from typing import List, Optional
from math import sqrt
import numpy as np

from .channel import ChannelGeometry, AxialDiscretization, ChannelInput, ChannelOutput
from .cooling import steam, getDT
from .correlations import get_correlation
from .friction import get_friction_model
from .waterflow import WaterFlow


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


def compute_mixed_outlet_temperature(
    temperatures: List[float],
    densities: List[float],
    specific_heats: List[float],
    flow_rates: List[float],
) -> float:
    """
    Compute energy-weighted mixed outlet temperature.

    Formula: T_out = Σ(Tᵢ·ρᵢ·cpᵢ·Q̇ᵢ) / Σ(ρᵢ·cpᵢ·Q̇ᵢ)

    Args:
        temperatures:  Outlet temperature per channel [K]
        densities:     Water density per channel [kg/m³]
        specific_heats: Specific heat per channel [J/kg/K]
        flow_rates:    Volumetric flow rate per channel [m³/s]

    Returns:
        Energy-weighted mixed outlet temperature [K]
    """
    Tout = 0.0
    rhoCpQ = 0.0
    for T, rho, cp, Q in zip(temperatures, densities, specific_heats, flow_rates):
        weight = rho * cp * Q
        Tout += T * weight
        rhoCpQ += weight
    return Tout / rhoCpQ


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

    # --- Inner velocity-iteration controls ---------------------------------
    # Maximum number of Darcy-Weisbach / friction-factor iterations.
    _VELOCITY_MAX_ITER: int = 10
    # Convergence threshold applied to both relative velocity and relative
    # friction-factor changes: |1 - x_new/x_old| <= _VELOCITY_TOLERANCE.
    _VELOCITY_TOLERANCE: float = 1e-3
    # Starting friction factor used when no better estimate is available.
    _FRICTION_INITIAL_GUESS: float = 0.055
    # -----------------------------------------------------------------------

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

        correlation = get_correlation(inputs.heat_correlation, inputs.fuzzy_factor)
        friction_model = get_friction_model(inputs.friction_model)

        channel_outputs = []
        errors_temp = []
        errors_heat = []

        for i, channel in enumerate(inputs.channels):
            if self.verbose:
                print(f"\n=== Channel {i}: {channel.geometry.name} ===")

            if channel.axial_discretization:
                # gradHZ mode with axial discretization
                output = self._compute_channel_axial(channel, inputs, correlation, friction_model)
            else:
                # Standard mode (mean values)
                output = self._compute_channel_uniform(channel, inputs, correlation, friction_model)

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

    def _solve_velocity(
        self,
        state,
        dPw_Pa: float,
        geom,
        pextra: float,
        friction_model,
        U_init: float,
        f_init: Optional[float] = None,
    ) -> tuple:
        """
        Iteratively solve for water velocity from a known pressure drop.

        Uses the Darcy-Weisbach equation coupled with the chosen friction
        model.  Iteration stops when both the relative change in velocity
        and the relative change in friction factor fall below
        ``_VELOCITY_TOLERANCE``.

        Args:
            state:         Water thermodynamic state at mean temperature.
            dPw_Pa:        Pressure drop [Pa].
            geom:          Channel geometry (hydraulic_diameter, length).
            pextra:        Additional pressure-loss coefficient.
            friction_model: OOP friction model instance.
            U_init:        Initial velocity guess [m/s].
            f_init:        Initial friction-factor guess (default:
                           ``_FRICTION_INITIAL_GUESS``).

        Returns:
            Tuple ``(velocity [m/s], friction_factor)``.

        Raises:
            RuntimeError: If the loop does not converge within
                          ``_VELOCITY_MAX_ITER`` iterations.
        """
        U = U_init
        f = f_init if f_init is not None else self._FRICTION_INITIAL_GUESS

        for _ in range(self._VELOCITY_MAX_ITER):
            Re = state.density * U * geom.hydraulic_diameter / state.dynamic_viscosity
            nf = friction_model.compute(Re, geom.hydraulic_diameter, f)
            nU = sqrt(
                2.0 * dPw_Pa
                / (state.density * (pextra + nf * geom.length / geom.hydraulic_diameter))
            )
            err_U = abs(1 - nU / U) if U > 0 else 1.0
            err_f = abs(1 - nf / f) if f > 0 else 1.0
            U, f = nU, nf
            if err_U <= self._VELOCITY_TOLERANCE and err_f <= self._VELOCITY_TOLERANCE:
                return U, f

        raise RuntimeError(
            f"Velocity iteration did not converge within {self._VELOCITY_MAX_ITER} iterations"
        )

    def _compute_channel_uniform(
        self, channel: ChannelInput, global_inputs: ThermalHydraulicInput,
        correlation, friction_model,
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

        dPw_Pa = global_inputs.pressure_drop * 1.0e5  # bar → Pa
        pextra = global_inputs.extra_pressure_loss

        for iteration in range(global_inputs.max_iterations):
            # Current estimates
            T_mean = channel.temp_inlet + dT / 2.0
            flow = U * geom.cross_section

            # Compute new temperature rise
            dT_new = getDT(flow, channel.power, T_mean, global_inputs.pressure_inlet)

            # Compute new heat transfer coefficient via OOP correlation
            h_new = correlation.compute(
                T_mean, global_inputs.pressure_inlet, U,
                geom.hydraulic_diameter, geom.length,
            )

            # Compute new velocity from pressure drop via OOP friction model
            state = steam(T_mean, global_inputs.pressure_inlet)
            U_new, cf_new = self._solve_velocity(
                state, dPw_Pa, geom, pextra, friction_model, U, cf
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
            density_outlet=Steam_outlet.density,
            specific_heat_outlet=Steam_outlet.specific_heat,
            converged=converged,
            iterations=iteration + 1,
        )

    def _compute_channel_axial(
        self, channel: ChannelInput, global_inputs: ThermalHydraulicInput,
        correlation, friction_model,
    ) -> ChannelOutput:
        """Compute channel with axial discretization (gradHZ mode)"""

        geom = channel.geometry
        axial = channel.axial_discretization
        n_sections = axial.n_sections

        # Initialize distributions
        T_z = [channel.temp_inlet] + [channel.temp_inlet + 10.0] * n_sections
        h_z = [80000.0] * (n_sections + 1)

        # Initial velocity guess
        U = channel.velocity_guess if channel.velocity_guess is not None else 5.0

        converged = False
        iteration = 0

        dPw_Pa = global_inputs.pressure_drop * 1.0e5  # bar → Pa
        pextra = global_inputs.extra_pressure_loss
        z_span = axial.z_positions[-1] - axial.z_positions[0]

        for iteration in range(global_inputs.max_iterations):
            flow = U * geom.cross_section
            T_z_old = T_z.copy()

            # Compute temperature distribution
            for k in range(n_sections):
                # Local pressure
                z_frac = (axial.z_positions[k] - axial.z_positions[0]) / z_span
                P_local = global_inputs.pressure_inlet - global_inputs.pressure_drop * z_frac

                # Temperature rise in this section
                T_mean_section = (T_z[k] + T_z[k + 1]) / 2.0
                dT_section = getDT(flow, axial.power_distribution[k], T_mean_section, P_local)
                T_z[k + 1] = T_z[k] + dT_section

            # Compute heat coefficient distribution via OOP correlation
            for k in range(len(T_z)):
                z_frac = (axial.z_positions[k] - axial.z_positions[0]) / z_span
                P_local = global_inputs.pressure_inlet - global_inputs.pressure_drop * z_frac

                h_z[k] = correlation.compute(
                    T_z[k], P_local, U, geom.hydraulic_diameter, geom.length,
                )

            # Update velocity via OOP friction model
            T_mean = (T_z[0] + T_z[-1]) / 2.0
            state = steam(T_mean, global_inputs.pressure_inlet)
            U_old = U
            U, cf = self._solve_velocity(
                state, dPw_Pa, geom, pextra, friction_model, U
            )

            # Check convergence
            err_flow = abs(1 - U / U_old) if U_old > 0 else 1.0
            err_temp = max(abs(1 - T_new / T_old) for T_new, T_old in zip(T_z[1:], T_z_old[1:]))

            if self.verbose:
                print(
                    f"  Iter {iteration}: U={U:.3f}, T_out={T_z[-1]:.3f}, "
                    f"err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
                )

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
            density_outlet=Steam_outlet.density,
            specific_heat_outlet=Steam_outlet.specific_heat,
            converged=converged,
            iterations=iteration + 1,
        )

    def _compute_mixed_outlet_temp(self, channels: List[ChannelOutput], pressure: float) -> float:
        """Compute mixed outlet temperature from all channels"""

        return compute_mixed_outlet_temperature(
            temperatures=[ch.temp_outlet for ch in channels],
            densities=[ch.density_outlet for ch in channels],
            specific_heats=[ch.specific_heat_outlet for ch in channels],
            flow_rates=[ch.flow_rate for ch in channels],
        )

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
