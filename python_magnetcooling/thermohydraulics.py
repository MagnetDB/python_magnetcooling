"""
Thermal-hydraulic calculations for cooling channels.

This module is independent of FeelPP and can be used for any
water-cooled magnet thermal analysis.

Six cooling levels are supported (see :class:`CoolingLevel`):

* ``mean``    – calorimetric ΔT, global h, U from pump curve (non-iterative).
* ``meanH``   – same per channel, flow ∝ Sh.
* ``grad``    – U from ΔP friction equation, global h (iterative).
* ``gradH``   – same per channel.
* ``gradHZ``  – per channel + axial power distribution, one mean h per channel.
* ``gradHZH`` – same, one h per axial section.
"""

from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Tuple

import numpy as np

# Public dataclasses live in channel.py (single source of truth).
from .channel import (
    AxialDiscretization,
    ChannelGeometry,
    ChannelInput,
    ChannelOutput,
    CoolingLevel,
)
from .correlations import HeatCorrelation, get_correlation
from .friction import FrictionModel, get_friction_model
from .water_properties import WaterProperties, WaterState
from .waterflow import WaterFlow

# getDT is a thin calorimetric helper; keep the import rather than inlining.
from .cooling import getDT


# ---------------------------------------------------------------------------
# Input / Output structures
# ---------------------------------------------------------------------------


@dataclass
class ThermalHydraulicInput:
    """Complete input for thermal-hydraulic analysis."""

    channels: List[ChannelInput]
    pressure_inlet: float  # bar
    pressure_drop: float  # bar

    # Which physical model to run
    cooling_level: CoolingLevel = CoolingLevel.GRAD

    # Required for mean / meanH: total volumetric flow rate from the pump curve
    # at the operating current.  Must be None for grad* levels.
    total_flow_rate: Optional[float] = None  # m³/s

    # Heat-transfer / friction parameters
    heat_correlation: str = "Montgomery"  # Montgomery | Dittus | Colburn | Silverberg
    friction_model: str = "Constant"  # Constant | Blasius | Filonenko | Colebrook | Swamee
    fuzzy_factor: float = 1.0
    extra_pressure_loss: float = 1.0  # dimensionless Pextra coefficient

    # Iterative solver settings (ignored for mean / meanH)
    max_iterations: int = 10
    tolerance_flow: float = 1e-3
    tolerance_temp: float = 1e-3
    relaxation_factor: float = 0.0  # 0 = no relaxation, 1 = full old value


@dataclass
class ThermalHydraulicOutput:
    """Complete thermal-hydraulic analysis results."""

    channels: List[ChannelOutput]

    # Global aggregates
    total_flow_rate: float  # m³/s
    outlet_temp_mixed: float  # K, energy-weighted mixed outlet
    total_power: float  # W

    # Convergence diagnostics
    max_error_temp: float = 0.0
    max_error_heat_coeff: float = 0.0
    converged: bool = True


# ---------------------------------------------------------------------------
# Module-level helper (also re-exported via cooling.getTout)
# ---------------------------------------------------------------------------


def compute_mixed_outlet_temperature(
    temperatures: List[float],
    densities: List[float],
    specific_heats: List[float],
    flow_rates: List[float],
) -> float:
    """
    Energy-weighted mixed outlet temperature.

    Formula: T_out = Σ(Tᵢ·ρᵢ·cpᵢ·Q̇ᵢ) / Σ(ρᵢ·cpᵢ·Q̇ᵢ)

    Args:
        temperatures:   Outlet temperature per channel [K]
        densities:      Water density per channel [kg/m³]
        specific_heats: Specific heat per channel [J/kg/K]
        flow_rates:     Volumetric flow rate per channel [m³/s]

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


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------


class ThermalHydraulicCalculator:
    """
    Standalone thermal-hydraulic calculator for water-cooled magnets.

    Supports six cooling levels via :class:`CoolingLevel`.  The level is
    carried in :class:`ThermalHydraulicInput` and controls which internal
    solver method is dispatched.

    Example (grad scenario, single channel)::

        calc = ThermalHydraulicCalculator()
        channel = ChannelInput(
            geometry=ChannelGeometry(
                hydraulic_diameter=0.008, cross_section=5e-5, length=0.5
            ),
            power=50_000,
            temp_inlet=290.0,
        )
        inputs = ThermalHydraulicInput(
            channels=[channel],
            pressure_inlet=15.0,
            pressure_drop=5.0,
            cooling_level=CoolingLevel.GRAD,
        )
        result = calc.compute(inputs)
        print(f"Outlet: {result.channels[0].temp_outlet:.2f} K")
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        # Set during compute(); kept as attributes so sub-methods can access
        # them without threading them through every signature.
        self._correlation: Optional[HeatCorrelation] = None
        self._friction: Optional[FrictionModel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, inputs: ThermalHydraulicInput) -> ThermalHydraulicOutput:
        """
        Compute thermal-hydraulic solution for all channels.

        Args:
            inputs: Complete input specification.

        Returns:
            Complete thermal-hydraulic solution.

        Raises:
            ValueError: If inputs are invalid for the chosen cooling level.
            RuntimeError: If the iterative solver does not converge.
        """
        self._validate_inputs(inputs)

        # Instantiate OOP correlation and friction model once per run.
        self._correlation = get_correlation(inputs.heat_correlation, inputs.fuzzy_factor)
        self._friction = get_friction_model(inputs.friction_model)

        # Pre-distribute flow per channel (only meaningful for mean/meanH).
        channel_flows = self._distribute_flow(inputs)

        channel_outputs: List[ChannelOutput] = []
        errors_temp: List[float] = []
        errors_heat: List[float] = []

        for i, (channel, Q_i) in enumerate(zip(inputs.channels, channel_flows)):
            if self.verbose:
                print(f"\n=== Channel {i}: {channel.geometry.name} ===")

            level = inputs.cooling_level
            if level.is_mean:
                output = self._compute_channel_mean(channel, inputs, Q_i)
            elif level.is_axial:
                output = self._compute_channel_axial(channel, inputs)
            else:
                output = self._compute_channel_uniform(channel, inputs)

            channel_outputs.append(output)

            if channel.temp_outlet_guess is not None:
                errors_temp.append(abs(1 - channel.temp_outlet_guess / output.temp_outlet))
            if channel.heat_coeff_guess is not None:
                errors_heat.append(abs(1 - channel.heat_coeff_guess / output.heat_coeff))

        outlet_temp_mixed = self._compute_mixed_outlet_temp(channel_outputs)
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
        self,
        inputs: ThermalHydraulicInput,
        waterflow_params: WaterFlow,
        current: float,
    ) -> ThermalHydraulicOutput:
        """
        Compute thermal-hydraulics using waterflow pump characteristics.

        Overrides pressure and (for mean/meanH) flow-rate values in *inputs*
        with values derived from the pump curve at *current*.

        Args:
            inputs:            Thermal-hydraulic inputs.
            waterflow_params:  Pump characteristic curves.
            current:           Operating current [A].

        Returns:
            Complete thermal-hydraulic solution.
        """
        inputs.pressure_inlet = waterflow_params.pressure(current)
        inputs.pressure_drop = waterflow_params.pressure_drop(current)

        if inputs.cooling_level.is_mean:
            # Total volumetric flow rate directly from pump curve [m³/s].
            inputs.total_flow_rate = waterflow_params.flow_rate(current)
        else:
            # Provide a velocity hint for faster convergence of the grad solver.
            total_Sh = sum(ch.geometry.cross_section for ch in inputs.channels)
            Q_total = waterflow_params.flow_rate(current)
            U_hint = Q_total / total_Sh
            for channel in inputs.channels:
                if channel.velocity_guess is None:
                    channel.velocity_guess = U_hint

        return self.compute(inputs)

    # ------------------------------------------------------------------
    # Flow distribution
    # ------------------------------------------------------------------

    def _distribute_flow(self, inputs: ThermalHydraulicInput) -> List[Optional[float]]:
        """
        Return per-channel volumetric flow rates [m³/s].

        * mean / meanH: Sh-proportional split of total_flow_rate.
        * grad*:        Returns [None, …]; each channel solves its own U from ΔP.
        """
        if not inputs.cooling_level.is_mean:
            return [None] * len(inputs.channels)

        total_Sh = sum(ch.geometry.cross_section for ch in inputs.channels)
        return [
            inputs.total_flow_rate * ch.geometry.cross_section / total_Sh
            for ch in inputs.channels
        ]

    # ------------------------------------------------------------------
    # Velocity solver (replaces cooling.Uw; uses OOP friction model)
    # ------------------------------------------------------------------

    def _solve_velocity(
        self,
        pressure_drop: float,
        hydraulic_diameter: float,
        length: float,
        state: WaterState,
        extra_pressure: float,
        velocity_guess: float = 5.0,
    ) -> Tuple[float, float]:
        """
        Solve flow velocity from pressure drop using the OOP friction model.

        ΔP = [Pextra + f·L/Dh] · ρ/2 · U²

        Args:
            pressure_drop:      Pressure drop [bar].
            hydraulic_diameter: Dh [m].
            length:             Channel length [m].
            state:              Water thermodynamic state.
            extra_pressure:     Dimensionless extra-loss coefficient Pextra.
            velocity_guess:     Initial U [m/s].

        Returns:
            (velocity [m/s], friction_factor [dimensionless])
        """
        U = max(velocity_guess, 1e-3)
        f = 0.055
        dP_Pa = pressure_drop * 1e5

        for _ in range(10):
            Re = state.density * U * hydraulic_diameter / state.dynamic_viscosity
            f_new = self._friction.compute(Re, hydraulic_diameter, f)
            U_new = sqrt(
                2.0 * dP_Pa / (state.density * (extra_pressure + f_new * length / hydraulic_diameter))
            )
            err_U = abs(1.0 - U_new / U)
            err_f = abs(1.0 - f_new / f)
            U, f = U_new, f_new
            if err_U < 1e-3 and err_f < 1e-3:
                return U, f

        return U, f  # return best estimate even if tolerance not met

    # ------------------------------------------------------------------
    # Channel solvers
    # ------------------------------------------------------------------

    def _compute_channel_mean(
        self,
        channel: ChannelInput,
        inputs: ThermalHydraulicInput,
        Q_i: float,
    ) -> ChannelOutput:
        """
        Non-iterative solver for ``mean`` / ``meanH`` cooling levels.

        U is fixed from the pre-distributed flow rate Q_i.
        ΔT is computed calorimetrically (one correction step).
        h is evaluated once at the mean temperature with fixed U.
        """
        geom = channel.geometry
        U = Q_i / geom.cross_section
        T_in = channel.temp_inlet
        P = inputs.pressure_inlet

        # Calorimetric ΔT: one correction step (compute at T_in, refine at T_mean).
        dT = getDT(Q_i, channel.power, T_in, P)
        T_mean = T_in + dT / 2.0
        dT = getDT(Q_i, channel.power, T_mean, P)  # refine with better T_mean
        T_out = T_in + dT
        T_mean = T_in + dT / 2.0

        # Heat coefficient at mean temperature with fixed U (no pressure-drop solve).
        h = self._correlation.compute(T_mean, P, U, geom.hydraulic_diameter, geom.length)

        # Friction factor is informational (U is not derived from it here).
        state_mean = WaterProperties.get_state(T_mean, P)
        Re = state_mean.density * U * geom.hydraulic_diameter / state_mean.dynamic_viscosity
        cf = self._friction.compute(Re, geom.hydraulic_diameter)

        state_out = WaterProperties.get_state(T_out, P)

        if self.verbose:
            print(
                f"  [mean] U={U:.3f} m/s, dT={dT:.3f} K, h={h:.1f} W/m²/K, Re={Re:.0f}"
            )

        return ChannelOutput(
            velocity=U,
            flow_rate=Q_i,
            friction_factor=cf,
            temp_inlet=T_in,
            temp_outlet=T_out,
            temp_rise=dT,
            temp_mean=T_mean,
            heat_coeff=h,
            density_outlet=state_out.density,
            specific_heat_outlet=state_out.specific_heat,
            converged=True,
            iterations=1,
        )

    def _compute_channel_uniform(
        self,
        channel: ChannelInput,
        inputs: ThermalHydraulicInput,
    ) -> ChannelOutput:
        """
        Iterative solver for ``grad`` / ``gradH`` cooling levels.

        U is solved self-consistently from the pressure-drop equation.
        ΔT and h converge together with U.
        """
        geom = channel.geometry

        U = channel.velocity_guess if channel.velocity_guess is not None else 5.0
        dT = (
            channel.temp_outlet_guess - channel.temp_inlet
            if channel.temp_outlet_guess is not None
            else 10.0
        )
        h = channel.heat_coeff_guess if channel.heat_coeff_guess is not None else 80_000.0
        cf = 0.055
        converged = False
        iteration = 0

        for iteration in range(inputs.max_iterations):
            T_mean = channel.temp_inlet + dT / 2.0
            Q = U * geom.cross_section

            dT_new = getDT(Q, channel.power, T_mean, inputs.pressure_inlet)

            h_new = self._correlation.compute(
                T_mean,
                inputs.pressure_inlet,
                U,
                geom.hydraulic_diameter,
                geom.length,
            )

            state = WaterProperties.get_state(T_mean, inputs.pressure_inlet)
            U_new, cf_new = self._solve_velocity(
                inputs.pressure_drop,
                geom.hydraulic_diameter,
                geom.length,
                state,
                inputs.extra_pressure_loss,
                U,
            )

            Q_new = U_new * geom.cross_section
            err_flow = abs(1.0 - Q_new / Q) if Q > 0 else 1.0
            err_temp = abs(1.0 - dT_new / dT) if abs(dT) > 0 else 1.0

            if self.verbose:
                print(
                    f"  Iter {iteration}: U={U_new:.3f}, dT={dT_new:.3f}, "
                    f"h={h_new:.1f}, err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
                )

            relax = inputs.relaxation_factor
            U = (1 - relax) * U_new + relax * U
            dT = (1 - relax) * dT_new + relax * dT
            h = (1 - relax) * h_new + relax * h
            cf = cf_new

            if err_flow < inputs.tolerance_flow and err_temp < inputs.tolerance_temp:
                converged = True
                break

        T_out = channel.temp_inlet + dT
        state_out = WaterProperties.get_state(T_out, inputs.pressure_inlet)

        return ChannelOutput(
            velocity=U,
            flow_rate=U * geom.cross_section,
            friction_factor=cf,
            temp_inlet=channel.temp_inlet,
            temp_outlet=T_out,
            temp_rise=dT,
            temp_mean=channel.temp_inlet + dT / 2.0,
            heat_coeff=h,
            density_outlet=state_out.density,
            specific_heat_outlet=state_out.specific_heat,
            converged=converged,
            iterations=iteration + 1,
        )

    def _compute_channel_axial(
        self,
        channel: ChannelInput,
        inputs: ThermalHydraulicInput,
    ) -> ChannelOutput:
        """
        Axial-marching solver for ``gradHZ`` / ``gradHZH`` cooling levels.

        Temperature is marched section by section along z.
        Heat coefficient is computed per section (at section midpoint T).

        ``gradHZ``  → ``heat_coeff`` = mean of section values; distribution not stored.
        ``gradHZH`` → ``heat_coeff_distribution`` stores all section values for feelpp.
        """
        geom = channel.geometry
        axial = channel.axial_discretization
        n = axial.n_sections
        per_section_h = inputs.cooling_level.has_per_section_h

        # Initial guesses: T at each section boundary (n+1 values).
        T_z = [channel.temp_inlet] + [channel.temp_inlet + 10.0] * n
        # h per section (midpoint); n values.
        h_z = [80_000.0] * n

        U = channel.velocity_guess if channel.velocity_guess is not None else 5.0
        cf = 0.055
        converged = False
        iteration = 0

        z0 = axial.z_positions[0]
        z_span = axial.z_positions[-1] - z0

        for iteration in range(inputs.max_iterations):
            Q = U * geom.cross_section
            T_z_old = T_z.copy()

            # --- Temperature marching along z --------------------------------
            for k in range(n):
                z_frac = (axial.z_positions[k] - z0) / z_span
                P_local = inputs.pressure_inlet - inputs.pressure_drop * z_frac

                # Use current T_z[k+1] as a mean-temperature guess.
                T_mean_k = (T_z[k] + T_z[k + 1]) / 2.0
                dT_k = getDT(Q, axial.power_distribution[k], T_mean_k, P_local)
                T_z[k + 1] = T_z[k] + dT_k

            # --- Heat coefficient per section --------------------------------
            for k in range(n):
                T_mid_k = (T_z[k] + T_z[k + 1]) / 2.0
                z_frac = (axial.z_positions[k] - z0) / z_span
                P_local = inputs.pressure_inlet - inputs.pressure_drop * z_frac
                h_z[k] = self._correlation.compute(
                    T_mid_k,
                    P_local,
                    U,
                    geom.hydraulic_diameter,
                    geom.length,
                )

            # --- Update velocity at channel mean temperature -----------------
            T_mean = (T_z[0] + T_z[-1]) / 2.0
            state = WaterProperties.get_state(T_mean, inputs.pressure_inlet)
            U_new, cf = self._solve_velocity(
                inputs.pressure_drop,
                geom.hydraulic_diameter,
                geom.length,
                state,
                inputs.extra_pressure_loss,
                U,
            )

            err_flow = abs(1.0 - U_new / U) if U > 0 else 1.0
            err_temp = max(
                abs(1.0 - T_z[k + 1] / T_z_old[k + 1])
                for k in range(n)
                if T_z_old[k + 1] != 0.0
            )

            if self.verbose:
                print(
                    f"  Iter {iteration}: U={U_new:.3f}, T_out={T_z[-1]:.3f}, "
                    f"err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
                )

            U = U_new

            if err_flow < inputs.tolerance_flow and err_temp < inputs.tolerance_temp:
                converged = True
                break

        state_out = WaterProperties.get_state(T_z[-1], inputs.pressure_inlet)

        # Per-section temperature rises: dTw_k = T_z[k+1] - T_z[k]  (n values).
        # Together with temp_inlet these give feelpp the local Tw at each section.
        dTw_sections = [T_z[k + 1] - T_z[k] for k in range(n)]

        return ChannelOutput(
            velocity=U,
            flow_rate=U * geom.cross_section,
            friction_factor=cf,
            temp_inlet=T_z[0],
            temp_outlet=T_z[-1],
            temp_rise=T_z[-1] - T_z[0],
            temp_mean=float(np.mean(T_z)),
            heat_coeff=float(np.mean(h_z)),
            # gradHZH: per-section h for feelpp boundary conditions.
            heat_coeff_distribution=list(h_z) if per_section_h else None,
            # All axial levels: boundary temperatures and per-section rises.
            temp_distribution=list(T_z),
            temp_rise_distribution=dTw_sections,
            density_outlet=state_out.density,
            specific_heat_outlet=state_out.specific_heat,
            converged=converged,
            iterations=iteration + 1,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_mixed_outlet_temp(self, channels: List[ChannelOutput]) -> float:
        return compute_mixed_outlet_temperature(
            temperatures=[ch.temp_outlet for ch in channels],
            densities=[ch.density_outlet for ch in channels],
            specific_heats=[ch.specific_heat_outlet for ch in channels],
            flow_rates=[ch.flow_rate for ch in channels],
        )

    def _validate_inputs(self, inputs: ThermalHydraulicInput) -> None:
        if not inputs.channels:
            raise ValueError("At least one channel is required.")
        if inputs.pressure_inlet <= 0:
            raise ValueError("Inlet pressure must be positive.")
        if inputs.pressure_drop < 0:
            raise ValueError("Pressure drop cannot be negative.")

        if inputs.cooling_level.is_mean and inputs.total_flow_rate is None:
            raise ValueError(
                f"CoolingLevel.{inputs.cooling_level.name} requires total_flow_rate to be set."
            )
        if inputs.cooling_level.is_mean and inputs.total_flow_rate <= 0:
            raise ValueError("total_flow_rate must be positive.")

        if inputs.cooling_level.is_axial:
            for i, ch in enumerate(inputs.channels):
                if ch.axial_discretization is None:
                    raise ValueError(
                        f"Channel {i} ({ch.geometry.name}): axial_discretization is required "
                        f"for CoolingLevel.{inputs.cooling_level.name}."
                    )

        for i, ch in enumerate(inputs.channels):
            if ch.power < 0:
                raise ValueError(f"Channel {i}: power cannot be negative.")
            if ch.temp_inlet <= 0:
                raise ValueError(f"Channel {i}: inlet temperature must be positive (K).")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


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
    cooling_level: CoolingLevel = CoolingLevel.GRAD,
    total_flow_rate: Optional[float] = None,
    verbose: bool = False,
) -> ChannelOutput:
    """
    Quick thermal-hydraulic calculation for a single channel.

    Args:
        hydraulic_diameter: Dh [m]
        cross_section:      Flow cross-section Sh [m²]
        length:             Channel length [m]
        power:              Power dissipated [W]
        temp_inlet:         Inlet temperature [K]
        pressure_inlet:     Inlet pressure [bar]
        pressure_drop:      Pressure drop [bar]
        heat_correlation:   Heat transfer correlation name
        friction_model:     Friction factor model name
        cooling_level:      Cooling model level (default: GRAD)
        total_flow_rate:    Required when cooling_level is MEAN [m³/s]
        verbose:            Print convergence info

    Returns:
        Channel thermal-hydraulic solution.
    """
    calc = ThermalHydraulicCalculator(verbose=verbose)

    channel = ChannelInput(
        geometry=ChannelGeometry(
            hydraulic_diameter=hydraulic_diameter,
            cross_section=cross_section,
            length=length,
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
        cooling_level=cooling_level,
        total_flow_rate=total_flow_rate,
    )

    result = calc.compute(inputs)
    return result.channels[0]
