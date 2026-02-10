#!/usr/bin/env python3
"""Heat exchanger calculations for primary cooling loop."""

from __future__ import annotations
from typing import Tuple, Optional

import math

import matplotlib

# Make LaTeX optional
try:
    matplotlib.rcParams["text.usetex"] = True
except Exception:
    pass  # LaTeX not available, use default rendering

import matplotlib.pyplot as plt
import pandas as pd
import ht

from .water_properties import WaterProperties
from .exceptions import (
    InvalidNTUError,
    InvalidHeatTransferError,
    InvalidTemperatureError,
)
from .heat_exchanger_config import HeatExchangerConfig, DEFAULT_HX_CONFIG


# Plotting helper function
def _create_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[dict],
    xlabel: str,
    ylabel: str,
    title: str = None,
    show: bool = False,
    save_path: str = None,
    grid: bool = True,
) -> None:
    """Create a standardized plot with common formatting.

    Args:
        df: DataFrame containing data
        x_col: Column name for x-axis
        y_cols: List of dicts with plot specs, e.g. [{'col': 'temp', 'color': 'blue', 'marker': 'o', ...}]
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title (optional)
        show: Display plot interactively
        save_path: Path to save plot (if not showing)
        grid: Show grid
    """
    fig, ax = plt.subplots()

    for y_spec in y_cols:
        col = y_spec.pop("col")
        df.plot(x=x_col, y=col, ax=ax, **y_spec)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if grid:
        plt.grid(True)

    if show:
        plt.show()
    elif save_path:
        plt.savefig(save_path, dpi=300)
        print(f"save to {save_path}")

    plt.close()


# Helper functions for backward compatibility with old water module API
def get_rho(pressure: float, temperature: float) -> float:
    """Get water density (kg/m³). Args: pressure (bar), temperature (°C)."""
    state = WaterProperties.get_state(temperature=temperature + 273.15, pressure=pressure)
    return state.density


def get_cp(pressure: float, temperature: float) -> float:
    """Get water specific heat (J/kg/K). Args: pressure (bar), temperature (°C)."""
    state = WaterProperties.get_state(temperature=temperature + 273.15, pressure=pressure)
    return state.specific_heat


def get_ohtc(
    mean_u_h: float,
    mean_u_c: float,
    de: float,
    p_hot: float,
    t_hot: float,
    p_cold: float,
    t_cold: float,
    params: list,
) -> float:
    """Calculate overall heat transfer coefficient.

    Args:
        mean_u_h: Mean velocity hot side (m/s)
        mean_u_c: Mean velocity cold side (m/s)
        de: Hydraulic diameter (m)
        p_hot: Hot side pressure (bar)
        t_hot: Hot side temperature (°C)
        p_cold: Cold side pressure (bar)
        t_cold: Cold side temperature (°C)
        params: Correlation parameters [a, b, c]

    Returns:
        Overall heat transfer coefficient (W/m²/K)
    """
    # Get water properties
    state_hot = WaterProperties.get_state(temperature=t_hot + 273.15, pressure=p_hot)
    state_cold = WaterProperties.get_state(temperature=t_cold + 273.15, pressure=p_cold)

    # Calculate Reynolds numbers
    re_hot = state_hot.density * mean_u_h * de / state_hot.dynamic_viscosity
    re_cold = state_cold.density * mean_u_c * de / state_cold.dynamic_viscosity

    # Correlation: Nu = a * Re^b * Pr^c
    a, b, c = params
    nu_hot = a * (re_hot**b) * (state_hot.prandtl**c)
    nu_cold = a * (re_cold**b) * (state_cold.prandtl**c)

    # Heat transfer coefficients
    h_hot = nu_hot * state_hot.thermal_conductivity / de
    h_cold = nu_cold * state_cold.thermal_conductivity / de

    # Overall heat transfer coefficient (assuming no wall resistance)
    return 1.0 / (1.0 / h_hot + 1.0 / h_cold)


def mixing_temp(flow1: float, p1: float, t1: float, flow2: float, p2: float, t2: float) -> float:
    """Compute mixing temperature of two water streams.

    Args:
        flow1: Flow rate stream 1 (m³/s)
        p1: Pressure stream 1 (bar)
        t1: Temperature stream 1 (°C)
        flow2: Flow rate stream 2 (m³/s)
        p2: Pressure stream 2 (bar)
        t2: Temperature stream 2 (°C)

    Returns:
        Mixed temperature (°C)
    """
    rho_cp_1 = get_rho(p1, t1) * get_cp(p1, t1)
    rho_cp_2 = get_rho(p2, t2) * get_cp(p2, t2)

    t_mix = (rho_cp_1 * flow1 * t1 + rho_cp_2 * flow2 * t2) / (rho_cp_1 * flow1 + rho_cp_2 * flow2)
    return t_mix


# Keep backward compatible name
mixingTemp = mixing_temp


def calculate_heat_profiles(
    df: pd.DataFrame,
    debit_alim: float,
    ohtc: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate heat transfer profiles for the cooling system.

    Args:
        df: DataFrame with required columns (teb, Thi, debitbrut, Flowhot, BP, Flow, Tout, HP, Tin, etc.)
        debit_alim: Cooling flow rate (m³/h)
        ohtc: Overall heat transfer coefficient (W/m²/K) or None to use computed values from df['Ohtc']

    Returns:
        DataFrame with calculated heat profiles
    """
    df = df.copy()

    def apply_heatexchange(row, ohtc_value):
        return (
            heatexchange(
                ohtc_value,
                row.teb,
                row.Thi,
                row.debitbrut / 3600.0,
                row.Flowhot,
                10,
                row.BP,
            )[2]
            / 1.0e6
        )

    if ohtc is not None:
        df["QNTU"] = df.apply(lambda row: apply_heatexchange(row, ohtc), axis=1)
    else:
        df["QNTU"] = df.apply(lambda row: apply_heatexchange(row, row.Ohtc), axis=1)

    df["Qhot"] = df.apply(
        lambda row: ((row.Flow) * 1.0e-3 + 0 / 3600.0)
        * (
            get_rho(row.BP, row.Tout) * get_cp(row.BP, row.Tout) * (row.Tout)
            - get_rho(row.HP, row.Tin) * get_cp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )

    df["QhotHx"] = df.apply(
        lambda row: (row.Flowhot)
        * (
            get_rho(row.BP, row.Thi) * get_cp(row.BP, row.Thi) * row.Thi
            - get_rho(row.HP, row.Tin) * get_cp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )

    df["QcoldHx"] = df.apply(
        lambda row: row.debitbrut
        / 3600.0
        * (
            get_rho(10, row.tsb) * get_cp(10, row.tsb) * row.tsb
            - get_rho(10, row.teb) * get_cp(10, row.teb) * row.teb
        )
        / 1.0e6,
        axis=1,
    )

    df["Pinstall"] = df.apply(
        lambda row: debit_alim
        / 3600.0
        * (
            get_rho(row.BP, row.TAlimout) * get_cp(row.BP, row.TAlimout) * row.TAlimout
            - get_rho(row.HP, row.Tin) * get_cp(row.HP, row.Tin) * row.Tin
        )
        / 1.0e6,
        axis=1,
    )

    df["Ppumps_Pinstall"] = df["Ptot"] - df["Pmagnet"]
    df["Ppumps"] = df["Ppumps_Pinstall"] - df["Pinstall"]

    return df


def plot_heat_balance(
    df: pd.DataFrame,
    experiment: str,
    ohtc: Optional[float],
    dT: float,
    show: bool = False,
) -> None:
    """Plot heat balance on magnet and heat exchanger sides.

    Args:
        df: DataFrame with heat profiles
        experiment: Experiment name
        ohtc: Heat transfer coefficient value or 'None'
        dT: Temperature difference
        show: If True, display plots interactively
    """
    # Plot installed power
    _create_plot(df, "t", [{"col": "Pinstall", "color": "yellow"}], r"t [s]", r"Q[MW]", show=show)

    # Plot magnet and total power
    _create_plot(df, "t", [{"col": "Pmagnet"}, {"col": "Ptot"}], r"t [s]", r"Q[MW]", show=show)

    # Plot pump power breakdown
    _create_plot(df, "t", [{"col": "Ppumps"}, {"col": "Pinstall"}], r"t [s]", r"Q[MW]", show=show)

    # Plot heat balance magnet side
    title = (
        f"HeatBalance Magnet side:{experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
        if ohtc is not None
        else f"HeatBalance Magnet side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
    )

    _create_plot(
        df,
        "t",
        [{"col": "Qhot"}, {"col": "Pmagnet"}, {"col": "Ptot"}],
        r"t [s]",
        r"Q[MW]",
        title=title,
        show=show,
    )

    # Plot heat balance HX side
    title = (
        f"HeatBalance HX side: {experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
        if ohtc is not None
        else f"HeatBalance HX side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
    )

    _create_plot(
        df,
        "t",
        [{"col": "QhotHx", "marker": "o", "markevery": 800, "alpha": 0.5}, {"col": "QcoldHx"}],
        r"t [s]",
        r"Q[MW]",
        title=title,
        show=show,
    )


def display_Q(
    inputfile: str,
    f_extension: str,
    df: pd.DataFrame,
    experiment: str,
    debit_alim: float,
    ohtc: Optional[float] = None,
    dT: float = 0.0,
    show: bool = False,
    extension: str = "-Q.png",
) -> None:
    """Calculate and plot heat profiles.

    Args:
        inputfile: Input file path
        f_extension: File extension
        df: DataFrame with required columns
        experiment: Experiment name
        debit_alim: Cooling flow rate (m³/h)
        ohtc: Heat transfer coefficient or None
        dT: Temperature difference
        show: Display plots interactively
        extension: Output file extension
    """
    # Calculate heat profiles
    df = calculate_heat_profiles(df, debit_alim, ohtc)

    # Plot results
    plot_heat_balance(df, experiment, ohtc, dT, show)

    # Save files if not showing
    if not show:
        title = (
            f"HeatBalance Magnet side:{experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
            if ohtc is not None
            else f"HeatBalance Magnet side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
        )

        imagefile = inputfile.replace(f_extension, "-Q_magnetside.png")
        _create_plot(
            df,
            "t",
            [{"col": "Qhot"}, {"col": "Pmagnet"}, {"col": "Ptot"}],
            r"t [s]",
            r"Q[MW]",
            title=title,
            save_path=imagefile,
        )

        title = (
            f"HeatBalance HX side: {experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
            if ohtc is not None
            else f"HeatBalance HX side: {experiment}: h=formula $W/m^2/K$, dT={dT}"
        )

        imagefile = inputfile.replace(f_extension, "-Q_hxside.png")
        _create_plot(
            df,
            "t",
            [{"col": "QhotHx", "marker": "o", "markevery": 800, "alpha": 0.5}, {"col": "QcoldHx"}],
            r"t [s]",
            r"Q[MW]",
            title=title,
            save_path=imagefile,
        )


def calculate_temperature_profiles(
    df: pd.DataFrame,
    tsb_key: str,
    tin_key: str,
    ohtc: Optional[float] = None,
    debug: bool = False,
) -> pd.DataFrame:
    """Calculate temperature profiles through heat exchanger.

    Args:
        df: DataFrame with required columns (teb, Thi, debitbrut, Flowhot, BP, Ohtc if ohtc is None)
        tsb_key: Column name for outlet cold side temperature
        tin_key: Column name for inlet hot side temperature
        ohtc: Heat transfer coefficient or None to use df['Ohtc']
        debug: Enable debug output

    Returns:
        DataFrame with temperature profiles
    """
    if debug:
        print("ohtc=", ohtc)

    df = df.copy()

    if debug:
        print(df.head())

    def apply_tsb(row, ohtc_value):
        return heatexchange(
            ohtc_value,
            row.teb,
            row.Thi,
            row.debitbrut / 3600.0,
            row.Flowhot,
            10,
            row.BP,
        )

    if ohtc is not None:
        df[[tsb_key, tin_key]] = df.apply(lambda row: pd.Series(apply_tsb(row, ohtc)[:2]), axis=1)
    else:
        df[[tsb_key, tin_key]] = df.apply(
            lambda row: pd.Series(apply_tsb(row, row.Ohtc)[:2]), axis=1
        )

    if debug:
        print(df[[tin_key, tsb_key]].head())

    return df


def plot_temperature_profiles(
    df: pd.DataFrame,
    tsb_key: str,
    tin_key: str,
    experiment: str,
    ohtc: Optional[float],
    dT: float,
    show: bool = False,
) -> None:
    """Plot temperature profiles.

    Args:
        df: DataFrame with temperature data
        tsb_key: Column name for outlet cold side temperature
        tin_key: Column name for inlet hot side temperature
        experiment: Experiment name
        ohtc: Heat transfer coefficient value
        dT: Temperature difference
        show: Display plot interactively
    """
    title = (
        f"{experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
        if ohtc is not None
        else f"{experiment}: h=computed $W/m^2/K$, dT={dT}"
    )

    _create_plot(
        df,
        "t",
        [
            {"col": tsb_key, "color": "blue", "marker": "o", "markevery": 800, "alpha": 0.5},
            {"col": "tsb", "color": "blue"},
            {"col": "teb", "color": "orange", "linestyle": "--"},
            {"col": tin_key, "color": "red", "marker": "o", "markevery": 800, "alpha": 0.5},
            {"col": "Tin", "color": "red"},
            {"col": "Tout", "color": "green", "linestyle": "--"},
            {"col": "cTout", "color": "yellow", "marker": "o", "markevery": 800, "alpha": 0.5},
            {"col": "Thi", "color": "brown", "marker": "x", "markevery": 800, "alpha": 0.5},
        ],
        r"t [s]",
        r"T[C]",
        title=title,
        show=show,
    )


def display_T(
    inputfile: str,
    f_extension: str,
    df: pd.DataFrame,
    experiment: str,
    tsb_key: str,
    tin_key: str,
    ohtc: Optional[float] = None,
    dT: float = 0.0,
    show: bool = False,
    extension: str = "-coolingloop.png",
    debug: bool = False,
) -> None:
    """Calculate and plot temperature profiles.

    Args:
        inputfile: Input file path
        f_extension: File extension
        df: DataFrame with required columns
        experiment: Experiment name
        tsb_key: Column name for outlet cold side temperature
        tin_key: Column name for inlet hot side temperature
        ohtc: Heat transfer coefficient or None
        dT: Temperature difference
        show: Display plots interactively
        extension: Output file extension
        debug: Enable debug output
    """
    if debug:
        print("ohtc=", ohtc)

    # Calculate temperature profiles
    df = calculate_temperature_profiles(df, tsb_key, tin_key, ohtc, debug)

    # Plot results
    plot_temperature_profiles(df, tsb_key, tin_key, experiment, ohtc, dT, show)

    # Save if not showing
    if not show:
        imagefile = inputfile.replace(f_extension, extension)
        title = (
            f"{experiment}: h={ohtc} $W/m^2/K$, dT={dT}"
            if ohtc is not None
            else f"{experiment}: h=computed $W/m^2/K$, dT={dT}"
        )

        _create_plot(
            df,
            "t",
            [
                {"col": tsb_key, "color": "blue", "marker": "o", "markevery": 800, "alpha": 0.5},
                {"col": "tsb", "color": "blue"},
                {"col": "teb", "color": "orange", "linestyle": "--"},
                {"col": tin_key, "color": "red", "marker": "o", "markevery": 800, "alpha": 0.5},
                {"col": "Tin", "color": "red"},
                {"col": "Tout", "color": "green", "linestyle": "--"},
                {"col": "cTout", "color": "yellow", "marker": "o", "markevery": 800, "alpha": 0.5},
                {"col": "Thi", "color": "brown", "marker": "x", "markevery": 800, "alpha": 0.5},
            ],
            r"t [s]",
            r"T[C]",
            title=title,
            save_path=imagefile,
        )


def estimate_temperature_elevation(
    power, flow_rate, inlet_temp, outlet_pressure, inlet_pressure, iterations=10
):
    """
    Estimate the temperature elevation of water in a pipe section where power is dissipated,
    accounting for temperature-dependent properties using the IAPWS97 package.

    Parameters:
    - power: Dissipated power in watts (W).
    - flow_rate: Volumetric flow rate in m^3/s.
    - inlet_temp: Inlet temperature in degrees Celsius.
    - outlet_pressure: Outlet pressure in bar.
    - inlet_pressure: Inlet pressure in bar.
    - iterations: Number of iterations for convergence.

    Returns:
    - outlet_temp: Outlet temperature in degrees Celsius.
    """
    eps = 1.0e-3
    inlet_temp_k = inlet_temp
    outlet_temp_k = inlet_temp_k

    for i in range(iterations):
        # Calculate properties at the current average temperature
        avg_temp_k = (inlet_temp_k + outlet_temp_k) / 2
        avg_pressure_k = (inlet_pressure + outlet_pressure) / 2

        rho = get_rho(pressure=avg_pressure_k, temperature=avg_temp_k)
        cp = get_cp(pressure=avg_pressure_k, temperature=avg_temp_k)

        # Mass flow rate (kg/s) from volumetric flow rate
        mass_flow_rate = rho * flow_rate

        # Recalculate temperature elevation
        delta_t = power / (mass_flow_rate * cp)

        # Update outlet temperature
        error = (outlet_temp_k - (inlet_temp_k + delta_t)) / outlet_temp_k
        # print("error=", error)
        outlet_temp_k = inlet_temp_k + delta_t
        if abs(error) <= eps:
            break

    # Convert outlet temperature back to Celsius
    outlet_temp = outlet_temp_k
    # print(
    #     f"Estimated outlet temperature:  {outlet_temp:.2f} °C after {i} iterations (error={error}, power={power:.2f} W, Tin={inlet_temp:.2f} C, Flow={flow_rate:.2f}) m/s."
    # )

    return outlet_temp


# def heatBalance(Tin, Pin, Debit, Power, debug=False):
#    """
#    Computes Tout from heatBalance
#
#    inputs:
#    Tin: input Temp in K
#    Pin: input Pressure (Bar)
#    Debit: Flow rate in kg/s
#    """
#
#    dT = Power / (w.getRho(Tin, Pin) * Debit * w.getCp(Tin, Pin))
#    Tout = Tin + dT
#    return Tout


def calculate_heat_capacity_and_density(
    Pci: float, Tci: float, Phi: float, Thi: float
) -> Tuple[float, float, float, float]:
    """Calculate heat capacity and density for hot and cold sides.

    Args:
        Pci: Cold side pressure (bar)
        Tci: Cold side temperature (°C)
        Phi: Hot side pressure (bar)
        Thi: Hot side temperature (°C)

    Returns:
        Tuple of (Cp_cold, Cp_hot, rho_hot, rho_cold)
    """
    Cp_cold = get_cp(Pci, Tci)  # J/kg/K
    Cp_hot = get_cp(Phi, Thi)  # J/kg/K
    rho_hot = get_rho(Phi, Thi)  # kg/m³
    rho_cold = get_rho(Pci, Tci)  # kg/m³
    return Cp_cold, Cp_hot, rho_hot, rho_cold


def calculate_mass_flow_rates(
    Debitc: float, Debith: float, rho_cold: float, rho_hot: float
) -> Tuple[float, float]:
    m_hot = rho_hot * Debith  # kg/s
    m_cold = rho_cold * Debitc  # kg/s
    return m_hot, m_cold


def validate_results(
    NTU: float,
    Q: float,
    Tco: float,
    Tho: float,
    h: float,
    Tci: float,
    Thi: float,
    Pci: float,
    Phi: float,
    Debitc: float,
    Debith: float,
) -> None:
    """Validate heat exchanger calculation results.

    Args:
        NTU: Number of Transfer Units
        Q: Heat transfer rate (W)
        Tco: Cold side outlet temperature (°C)
        Tho: Hot side outlet temperature (°C)
        h: Heat transfer coefficient (W/m²/K)
        Tci: Cold side inlet temperature (°C)
        Thi: Hot side inlet temperature (°C)
        Pci: Cold side pressure (bar)
        Phi: Hot side pressure (bar)
        Debitc: Cold side flow rate (m³/h)
        Debith: Hot side flow rate (l/s)

    Raises:
        InvalidNTUError: If NTU is infinite or NaN
        InvalidHeatTransferError: If heat transfer rate is infinite or NaN
        InvalidTemperatureError: If outlet temperatures are None
    """
    if NTU == float("inf") or math.isnan(NTU):
        raise InvalidNTUError(NTU, Tci, Thi, Pci, Phi, Debitc, Debith)

    if Q == float("inf") or math.isnan(Q):
        raise InvalidHeatTransferError(Q, Tci, Thi, Pci, Phi, Debitc, Debith)

    if Tco is None:
        raise InvalidTemperatureError("Tco", Tco, h, Tci, Thi, Pci, Phi, Debitc, Debith)

    if Tho is None:
        raise InvalidTemperatureError("Tho", Tho, h, Tci, Thi, Pci, Phi, Debitc, Debith)


def heatexchange(
    h: float,
    Tci: float,
    Thi: float,
    Debitc: float,
    Debith: float,
    Pci: float,
    Phi: float,
    debug: bool = False,
    hx_config: HeatExchangerConfig = None,
) -> Tuple[float, float, float]:
    """
    NTU Model for heat Exchanger

    compute the output temperature for the heat exchanger
    as a function of input temperatures and flow rates

    Tci: input Temp on cold side
    Thi: input Temp on hot side
    TA: output from cooling alim (on hot side)

    Debitc: m^3/h
    Debith: l/s
    hx_config: Heat exchanger configuration (uses default if None)
    """
    if hx_config is None:
        hx_config = DEFAULT_HX_CONFIG

    A = hx_config.area  # m^2
    Cp_cold, Cp_hot, rho_hot, rho_cold = calculate_heat_capacity_and_density(Pci, Tci, Phi, Thi)
    m_hot, m_cold = calculate_mass_flow_rates(Debitc, Debith, rho_cold, rho_hot)

    # For plate exchanger
    result = ht.hx.P_NTU_method(
        m_hot, m_cold, Cp_hot, Cp_cold, UA=h * A, T1i=Thi, T2i=Tci, subtype="1/1"
    )

    NTU = result["NTU1"]
    Q = result["Q"]
    Tco = result["T2o"]
    Tho = result["T1o"]

    validate_results(NTU, Q, Tco, Tho, h, Tci, Thi, Pci, Phi, Debitc, Debith)

    return Tco, Tho, Q


def calculate_extended_temperature_fields(
    df: pd.DataFrame,
    debit_alim: float,
) -> pd.DataFrame:
    """Calculate extended temperature fields including mixing temperatures.

    Args:
        df: DataFrame with basic temperature data
        debit_alim: Cooling flow rate (m³/h)

    Returns:
        DataFrame with additional temperature fields
    """

    # Calculate outlet temperatures for high and bitter sections
    def apply_ToutH(row):
        return estimate_temperature_elevation(row.PH, row.FlowH * 1.0e-3, row.TinH, row.BP, row.HPH)

    df["ToutH"] = df.apply(apply_ToutH, axis=1)

    def apply_ToutB(row):
        return estimate_temperature_elevation(row.PB, row.FlowB * 1.0e-3, row.TinB, row.BP, row.HPB)

    df["ToutB"] = df.apply(apply_ToutB, axis=1)

    # Calculate mixed outlet temperature
    def apply_mixingTemp(row):
        return mixing_temp(
            row.FlowH,
            row.BP,
            row.ToutH,
            row.FlowB,
            row.BP,
            row.ToutB,
        )

    df["cTout"] = df.apply(apply_mixingTemp, axis=1)

    # Calculate hot side inlet temperature with cooling flow
    def apply_mixingThi(row):
        return mixing_temp(
            row.Flow,
            row.BP,
            row.Tout,
            debit_alim / 3600.0,
            row.BP,
            row.TAlimout,
        )

    df["Thi"] = df.apply(apply_mixingThi, axis=1)

    return df


def calculate_heat_transfer_coefficients(
    df: pd.DataFrame,
    debit_alim: float,
    hx_config: HeatExchangerConfig = None,
) -> pd.DataFrame:
    """Calculate overall heat transfer coefficients.

    Args:
        df: DataFrame with flow and temperature data
        debit_alim: Cooling flow rate (m³/h)
        hx_config: Heat exchanger configuration (uses default if None)

    Returns:
        DataFrame with OHTC values
    """
    if hx_config is None:
        hx_config = DEFAULT_HX_CONFIG
    
    # Heat exchanger geometry from configuration
    Nc = hx_config.num_channels
    Ac = hx_config.channel_area
    de = hx_config.hydraulic_diameter
    cooling_params = hx_config.correlation_params

    # Calculate flow velocities
    df["Flowhot"] = df.apply(lambda row: ((row.Flow) * 1.0e-3 + debit_alim / 3600.0), axis=1)
    df["MeanU_h"] = df.apply(
        lambda row: (row.Flowhot) / (Ac * Nc),
        axis=1,
    )
    df["MeanU_c"] = df.apply(lambda row: (row.debitbrut / 3600.0) / (Ac * Nc), axis=1)

    # Calculate OHTC
    df["Ohtc"] = df.apply(
        lambda row: get_ohtc(
            row.MeanU_h,
            row.MeanU_c,
            de,
            row.BP,
            row.Thi,
            row.BP,
            row.teb,
            cooling_params,
        ),
        axis=1,
    )

    return df


def plot_temperature_comparison(
    df: pd.DataFrame,
    experiment: str,
    show: bool = False,
    save_path: str = None,
) -> None:
    """Plot comparison of measured vs calculated outlet temperatures.

    Args:
        df: DataFrame with temperature data
        experiment: Experiment name
        show: Display plot interactively
        save_path: Path to save plot (if not showing)
    """
    _create_plot(
        df,
        "t",
        [
            {"col": "Tout", "color": "blue", "marker": "o", "alpha": 0.5, "markevery": 800},
            {"col": "cTout", "color": "blue", "linestyle": "--"},
        ],
        r"t [s]",
        r"T [C]",
        title=f"{experiment}: Tout",
        show=show,
        save_path=save_path,
    )


def plot_mixed_temperatures(
    df: pd.DataFrame,
    show: bool = False,
) -> None:
    """Plot Tout, cTout, and Thi temperature evolution.

    Args:
        df: DataFrame with temperature data
        show: Display plot interactively
    """
    _create_plot(
        df, "t", [{"col": "Tout"}, {"col": "cTout"}, {"col": "Thi"}], r"t [s]", r"T[C]", show=show
    )


def plot_heat_transfer_coefficient(
    df: pd.DataFrame,
    experiment: str,
    show: bool = False,
    save_path: str = None,
) -> None:
    """Plot overall heat transfer coefficient evolution.

    Args:
        df: DataFrame with OHTC data
        experiment: Experiment name
        show: Display plot interactively
        save_path: Path to save plot (if not showing)
    """
    _create_plot(
        df,
        "t",
        [{"col": "Ohtc", "color": "red", "marker": "o", "markevery": 800, "alpha": 0.5}],
        r"t [s]",
        r"$W/m^2/K$",
        title=f"{experiment}: Heat Exchange Coefficient",
        show=show,
        save_path=save_path,
    )


def main(command_line: Optional[list[str]] = None) -> None:
    """Main entry point for heat exchanger analysis.

    Note: This function is deprecated and kept for backward compatibility.
    For new code, use the individual calculation functions directly with DataFrames.
    The data loading, preprocessing, smoothing, and filtering should be handled externally.
    """
    raise NotImplementedError(
        "The main() function has been deprecated. "
        "Data loading, smoothing, and filtering should be handled externally. "
        "Use the individual calculation functions (calculate_heat_profiles, calculate_temperature_profiles, etc.) "
        "directly with pandas DataFrames."
    )


if __name__ == "__main__":
    print(
        "This module now provides heat exchanger calculation functions only.\n"
        "Data loading and preprocessing should be handled externally.\n"
        "\n"
        "Example usage:\n"
        "  import pandas as pd\n"
        "  from python_magnetcooling.heatexchanger_primary import (\n"
        "      calculate_heat_profiles,\n"
        "      calculate_temperature_profiles,\n"
        "      calculate_extended_temperature_fields,\n"
        "      calculate_heat_transfer_coefficients\n"
        "  )\n"
        "  from python_magnetcooling.heat_exchanger_config import HeatExchangerConfig\n"
        "\n"
        "  # Load your data into a DataFrame\n"
        "  df = pd.read_csv('your_data.csv')\n"
        "\n"
        "  # Create heat exchanger configuration\n"
        "  hx_config = HeatExchangerConfig()  # Uses default student-fitted parameters\n"
        "  # Or use nominal parameters:\n"
        "  # hx_config = HeatExchangerConfig(use_nominal_params=True)\n"
        "\n"
        "  # Calculate fields\n"
        "  df = calculate_extended_temperature_fields(df, debit_alim=60)\n"
        "  df = calculate_heat_transfer_coefficients(df, debit_alim=60, hx_config=hx_config)\n"
        "  df = calculate_heat_profiles(df, debit_alim=60, ohtc=4000)\n"
    )
