"""
Cooling Models - Legacy Interface

This module provides a legacy function-based interface for thermal-hydraulic
cooling calculations. Every function here is a thin wrapper that delegates to
the newer class-based modules:

- Water properties: :mod:`water_properties` (WaterProperties, WaterState)
- Heat correlations: :mod:`correlations` (HeatCorrelation subclasses)
- Friction models:   :mod:`friction`      (FrictionModel subclasses)

New code should use those modules directly rather than this legacy interface.
"""

from typing import List
from math import sqrt

from .water_properties import WaterProperties, WaterState
from .correlations import (
    HeatCorrelation,
    MontgomeryCorrelation,
    DittusBoelterCorrelation,
    ColburnCorrelation,
    SilverbergCorrelation,
    get_correlation,
)
from .friction import (
    ConstantFriction,
    BlasiusFriction,
    FilonenkoFriction,
    ColebrookFriction,
    SwameeFriction,
    get_friction_model,
)


# ---------------------------------------------------------------------------
# Water properties helpers
# ---------------------------------------------------------------------------

def steam(Tw: float, P: float) -> WaterState:
    """
    Return water state object at given conditions.

    Delegates to WaterProperties.get_state().

    Args:
        Tw: Temperature [K]
        P:  Pressure [bar]

    Returns:
        WaterState with all thermodynamic properties
    """
    return WaterProperties.get_state(temperature=Tw, pressure=P)


def Reynolds(state: WaterState, U: float, Dh: float, L: float) -> float:
    """
    Compute Reynolds number: Re = ρ·U·Dh/μ

    Args:
        state: Water thermodynamic state (WaterState)
        U:     Flow velocity [m/s]
        Dh:    Hydraulic diameter [m]
        L:     Channel length [m]  (unused, kept for API compatibility)

    Returns:
        Reynolds number [dimensionless]
    """
    return state.density * U * Dh / state.dynamic_viscosity


def Prandtl(state: WaterState) -> float:
    """
    Return Prandtl number: Pr = μ·cp/k

    Args:
        state: Water thermodynamic state (WaterState)

    Returns:
        Prandtl number [dimensionless]
    """
    return state.prandtl


def Nusselt(params: tuple, Re: float, Pr: float) -> float:
    """
    Compute Nusselt number: Nu = α·Re^n·Pr^m

    Delegates to HeatCorrelation.compute_nusselt().

    Args:
        params: Tuple (α, n, m) — correlation coefficients
        Re:     Reynolds number
        Pr:     Prandtl number

    Returns:
        Nusselt number
    """
    alpha, n, m = params
    return HeatCorrelation.compute_nusselt(Re, Pr, alpha, n, m)


def getDT(flow: float, Power: float, Tw: float, P: float) -> float:
    """
    Compute temperature rise: ΔT = Power / (ρ·cp·flow)

    Delegates to WaterProperties.compute_temperature_rise().

    Args:
        flow:  Volumetric flow rate [m³/s]
        Power: Heat power [W]
        Tw:    Water temperature [K]
        P:     Water pressure [bar]

    Returns:
        Temperature rise [K]
    """
    return WaterProperties.compute_temperature_rise(
        flow_rate=flow, power=Power, temperature=Tw, pressure=P
    )


def getTout(T: List[float], VolMass: List[float], SpecHeat: List[float], Q: List[float]) -> float:
    """
    Compute energy-weighted mixed outlet temperature.

    Legacy wrapper — delegates to
    :func:`thermohydraulics.compute_mixed_outlet_temperature`.

    Args:
        T:        Outlet temperatures per channel [K]
        VolMass:  Water densities per channel [kg/m³]
        SpecHeat: Specific heats per channel [J/kg/K]
        Q:        Volumetric flow rates per channel [m³/s]

    Returns:
        Mixed outlet temperature [K]
    """
    # Lazy import avoids the circular dependency:
    # thermohydraulics imports cooling at module level, so cooling must not
    # import thermohydraulics at module level.
    from .thermohydraulics import compute_mixed_outlet_temperature
    return compute_mixed_outlet_temperature(T, VolMass, SpecHeat, Q)


# ---------------------------------------------------------------------------
# Heat transfer correlations — delegate to correlations.py OOP classes
# ---------------------------------------------------------------------------

def Montgomery(
    Tw: float,
    Pw: float,
    dPw: float,
    U: float,
    Dh: float,
    L: float,
    friction: str,
    fuzzy: float = 1.0,
    pextra: float = 1,
) -> float:
    """
    Compute heat transfer coefficient using Montgomery correlation [W/m²/K].

    Delegates to MontgomeryCorrelation.compute().

    Args:
        Tw:      Water temperature [K]
        Pw:      Water pressure [bar]
        dPw:     Pressure drop [bar]  (unused, kept for API compatibility)
        U:       Flow velocity [m/s]
        Dh:      Hydraulic diameter [m]
        L:       Channel length [m]
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:   Empirical correction factor
        pextra:  Extra pressure loss coefficient  (unused, kept for API compatibility)

    Reference: Montgomery, D.B. "Solenoid Magnet Design" (1969), p38 eq 3.3
    """
    return MontgomeryCorrelation(fuzzy_factor=fuzzy).compute(Tw, Pw, U, Dh, L)


def Dittus(
    Tw: float,
    Pw: float,
    dPw: float,
    U: float,
    Dh: float,
    L: float,
    friction: str,
    fuzzy: float = 1.0,
    pextra: float = 1,
) -> float:
    """
    Compute heat transfer coefficient using Dittus-Boelter correlation [W/m²/K].

    Delegates to DittusBoelterCorrelation.compute().

    Args:
        Tw:      Water temperature [K]
        Pw:      Water pressure [bar]
        dPw:     Pressure drop [bar]  (unused, kept for API compatibility)
        U:       Flow velocity [m/s]
        Dh:      Hydraulic diameter [m]
        L:       Channel length [m]
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:   Empirical correction factor
        pextra:  Extra pressure loss coefficient  (unused, kept for API compatibility)
    """
    return DittusBoelterCorrelation(fuzzy_factor=fuzzy).compute(Tw, Pw, U, Dh, L)


def Colburn(
    Tw: float,
    Pw: float,
    dPw: float,
    U: float,
    Dh: float,
    L: float,
    friction: str,
    fuzzy: float = 1.0,
    pextra: float = 1,
) -> float:
    """
    Compute heat transfer coefficient using Colburn correlation [W/m²/K].

    Delegates to ColburnCorrelation.compute().

    Args:
        Tw:      Water temperature [K]
        Pw:      Water pressure [bar]
        dPw:     Pressure drop [bar]  (unused, kept for API compatibility)
        U:       Flow velocity [m/s]
        Dh:      Hydraulic diameter [m]
        L:       Channel length [m]
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:   Empirical correction factor
        pextra:  Extra pressure loss coefficient  (unused, kept for API compatibility)
    """
    return ColburnCorrelation(fuzzy_factor=fuzzy).compute(Tw, Pw, U, Dh, L)


def Silverberg(
    Tw: float,
    Pw: float,
    dPw: float,
    U: float,
    Dh: float,
    L: float,
    friction: str,
    fuzzy: float = 1.0,
    pextra: float = 1,
) -> float:
    """
    Compute heat transfer coefficient using Silverberg correlation [W/m²/K].

    Delegates to SilverbergCorrelation.compute().

    Args:
        Tw:      Water temperature [K]
        Pw:      Water pressure [bar]
        dPw:     Pressure drop [bar]  (unused, kept for API compatibility)
        U:       Flow velocity [m/s]
        Dh:      Hydraulic diameter [m]
        L:       Channel length [m]
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:   Empirical correction factor
        pextra:  Extra pressure loss coefficient  (unused, kept for API compatibility)
    """
    return SilverbergCorrelation(fuzzy_factor=fuzzy).compute(Tw, Pw, U, Dh, L)


def getHeatCoeff(
    Dh: float,
    L: float,
    U: float,
    Tw: float,
    Pw: float,
    dPw: float,
    model: str = "Montgomery",
    friction: str = "Constant",
    fuzzy: float = 1.0,
    pextra: float = 1,
) -> float:
    """
    Compute heat transfer coefficient for the given correlation model [W/m²/K].

    Delegates to get_correlation(model, fuzzy).compute().

    Args:
        Dh:       Hydraulic diameter [m]
        L:        Channel length [m]
        U:        Flow velocity [m/s]
        Tw:       Water temperature [K]
        Pw:       Water pressure [bar]
        dPw:      Pressure drop [bar]  (unused, kept for API compatibility)
        model:    Correlation name (Montgomery, Dittus, Colburn, Silverberg)
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:    Empirical correction factor
        pextra:   Additional pressure loss coefficient  (unused, kept for API compatibility)

    Returns:
        Heat transfer coefficient [W/m²/K]
    """
    return get_correlation(model, fuzzy).compute(Tw, Pw, U, Dh, L)


# ---------------------------------------------------------------------------
# Friction factor models — delegate to friction.py OOP classes
# ---------------------------------------------------------------------------

def Constant(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """
    Return constant friction factor (0.055).

    Delegates to ConstantFriction.compute().
    """
    return ConstantFriction().compute(Re, Dh, f)


def Blasius(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """
    Blasius friction factor: f = 0.316/Re^0.25

    Delegates to BlasiusFriction.compute().
    """
    return BlasiusFriction().compute(Re, Dh, f)


def Filonenko(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """
    Filonenko friction factor: f = 1/(1.82·log₁₀(Re) - 1.64)²

    Delegates to FilonenkoFriction.compute().
    """
    return FilonenkoFriction().compute(Re, Dh, f)


def Colebrook(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """
    Colebrook-White friction factor (iterative).

    Delegates to ColebrookFriction.compute().
    """
    return ColebrookFriction(roughness=rugosity).compute(Re, Dh, f)


def Swamee(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """
    Swamee-Jain explicit approximation of Colebrook.

    Delegates to SwameeFriction.compute().
    """
    return SwameeFriction(roughness=rugosity).compute(Re, Dh, f)


def Uw(
    state: WaterState,
    dPw: float,
    Dh: float,
    L: float,
    friction: str = "Colebrook",
    Pextra: float = 1.0,
    fguess: float = 0.055,
    uguess: float = 0,
    rugosity: float = 0.012e-3,
) -> tuple:
    """
    Compute water velocity from pressure drop.

    Delegates friction-factor evaluation to get_friction_model(); the
    iterative Darcy-Weisbach velocity loop is unchanged.

    Args:
        state:    Water thermodynamic state (WaterState)
        dPw:      Pressure drop [bar]
        Dh:       Hydraulic diameter [m]
        L:        Channel length [m]
        friction: Friction model name
        Pextra:   Additional pressure loss coefficient
        fguess:   Initial friction factor guess
        uguess:   Initial velocity guess [m/s]
        rugosity: Surface roughness [m]

    Returns:
        Tuple (velocity [m/s], friction_factor)
    """
    friction_model = get_friction_model(friction, roughness=rugosity)

    U = uguess
    f = fguess
    dPw_Pascal = dPw * 1.0e5

    isOk = False
    for _ in range(10):
        Re = Reynolds(state, U, Dh, L)
        nf = friction_model.compute(Re, Dh, f)
        nU = sqrt(2 * dPw_Pascal / (state.density * (Pextra + nf * L / Dh)))
        error_U = abs(1 - nU / U) if U > 0 else 1.0
        error_f = abs(1 - nf / f) if f > 0 else 1.0
        U, f = nU, nf
        if error_U <= 1e-3 and error_f <= 1e-3:
            isOk = True
            break

    if not isOk:
        raise RuntimeError("Uw: max iterations reached without convergence")
    return U, f
