"""
Cooling Models - Legacy Interface

This module provides a legacy function-based interface for thermal-hydraulic
cooling calculations. It internally delegates to the newer class-based modules:

- Water properties: :mod:`water_properties` (WaterProperties, WaterState)
- Heat correlations: :mod:`correlations` (HeatCorrelation subclasses)
- Friction models: :mod:`friction` (FrictionModel subclasses)

New code should use these modules directly rather than this legacy interface.
"""

from typing import List
from math import exp, log, log10, sqrt

from .water_properties import WaterProperties, WaterState


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
    Compute heat exchange coefficient using Montgomery correlation [W/m²/K].

    Args:
        Tw:      Water temperature [K]
        Pw:      Water pressure [bar]  (unused, kept for API compatibility)
        dPw:     Pressure drop [bar]   (unused, kept for API compatibility)
        U:       Flow velocity [m/s]
        Dh:      Hydraulic diameter [m]
        L:       Channel length [m]    (unused, kept for API compatibility)
        friction: Friction model name  (unused, kept for API compatibility)
        fuzzy:   Empirical correction factor

    Note:
        Original Montgomery formula uses T in Celsius and Dh in centimeters.
        Here Dh is accepted in meters (coefficient adjusted from 0.1426 to
        1426.404 to account for the W/cm²/K → W/m²/K unit change; Dh
        remains in meters as documented).

    Reference: Montgomery, D.B. "Solenoid Magnet Design" (1969), p38 eq 3.3
    """
    h = fuzzy * 1426.404 * (1 + 1.5e-2 * (Tw - 273.15)) * exp(log(U) * 0.8) / exp(log(Dh) * 0.2)
    return h


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
    """Dittus-Boelter correlation: Nu = 0.023·Re^0.8·Pr^0.4 [W/m²/K]"""
    params = (0.023, 0.8, 0.4)
    h = hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, "Dittus")
    return h


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
    """Colburn correlation: Nu = 0.023·Re^0.8·Pr^0.3 [W/m²/K]"""
    params = (0.023, 0.8, 0.3)
    h = hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, "Colburn")
    return h


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
    """Silverberg correlation: Nu = 0.015·Re^0.85·Pr^0.3 [W/m²/K]"""
    params = (0.015, 0.85, 0.3)
    h = hcorrelation(params, Tw, Pw, dPw, U, Dh, L, friction, pextra, "Silverberg")
    return h


def Constant(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """Return constant friction factor (0.055)."""
    cf = 0.055
    return cf


"""
To be implemented

friction == "karman":
            iterate = True
            Cf  = math.pow(1.93*math.log10(Reynolds*math.sqrt(f))-0.537,-2)
        elif friction == "rough":
            iterate = True
            eps = 2.5e-2 # mm
            rstar = 1/math.sqrt(8.) * (Reynolds*math.sqrt(f))*eps/dh
            brstar = 1/(1.930*math.sqrt(f)) + math.log10(1.9/math.sqrt(8.) * eps/dh)
            ###print "brstar=%g" % brstar

            # Cf = math.pow(-1.930*math.log(1.90/(Reynolds*math.sqrt(f))*(1+0.34*rstar*math.exp(-11./rstar))),-2.)
            Cf = math.pow(-2.00*math.log10(2.51/(Reynolds*math.sqrt(f))*(1+rstar/3.3)),-2.)

        # Gnielinski breaks when a tends to 1
        # elif friction == "gnielinski":
        #     a = diameter_ratio
        #     Re = Reynolds * ( (1.+a**2) * math.log(a)+(1-a**2) / ( (1.-a)**2 * math.log(a) ))
        #     Cf = math.pow(1.8*math.log10(Re)-1.5,-2)
        # # print ("%s Cf=%g" % (friction,Cf) )
"""


def Blasius(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """Blasius friction factor: f = 0.316/Re^0.25"""
    cf = 0.316 / exp(log(Re) * 0.25)
    return cf


def Filonenko(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """Filonenko friction factor: f = 1/(1.82·log10(Re) - 1.64)²"""
    cf = 1 / (1.82 * log10(Re) - 1.64) ** 2
    return cf


def _iterative_convergence(
    initial_value: float,
    compute_new_value,
    max_iterations: int = 10,
    max_error: float = 1.0e-3,
    method_name: str = "Convergence",
) -> float:
    """Generic iterative convergence for friction factor calculations.

    Args:
        initial_value: Initial guess for the value
        compute_new_value: Function that computes new value from current value
        max_iterations: Maximum number of iterations
        max_error: Maximum relative error for convergence
        method_name: Name of the method (for error messages)

    Returns:
        Converged value

    Raises:
        RuntimeError: If convergence fails
    """
    val = initial_value
    isOk = False

    for it in range(max_iterations):
        nval = compute_new_value(val)
        error = abs(1 - nval / val)
        val = nval

        if error <= max_error:
            isOk = True
            break

    if not isOk:
        raise RuntimeError(f"{method_name}: failed to converge after {max_iterations} iterations")

    return val


def Colebrook(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """Colebrook-White friction factor (iterative)."""
    def compute_new(val):
        return -2 * log10(rugosity / (3.7 * Dh) + 2.51 / Re * val)

    val = _iterative_convergence(1 / sqrt(f), compute_new, method_name="Colebrook")
    cf = 1 / val**2
    return cf


def Swamee(Re: float, Dh: float, f: float, rugosity: float) -> float:
    """Swamee-Jain explicit approximation of Colebrook."""
    def compute_new(val):
        return 1.325 / log(rugosity / (3.7 * Dh) + 5.74 / exp(log(Re) * 0.9)) ** 2

    cf = _iterative_convergence(f, compute_new, method_name="Swamee")
    return cf


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
    friction_method = {
        "Constant": Constant,
        "Blasius": Blasius,
        "Filonenko": Filonenko,
        "Colebrook": Colebrook,
        "Swamee": Swamee,
    }

    U = uguess
    f = fguess

    isOk = False
    max_err_U = 1.0e-3
    max_err_f = 1.0e-3

    for it in range(10):
        Re = Reynolds(state, U, Dh, L)
        nf = friction_method[friction](Re, Dh, f, rugosity)

        dPw_Pascal = dPw * 1.0e5
        nU = sqrt(2 * dPw_Pascal / (state.density * (Pextra + nf * L / Dh)))
        error_U = abs(1 - nU / U)
        error_f = abs(1 - nf / f)
        U = nU
        f = nf

        if error_U <= max_err_U and error_f <= max_err_f:
            isOk = True
            break

    if not isOk:
        raise RuntimeError("Uw: max iterations reached without convergence")
    return U, f


def Nusselt(params: tuple, Re: float, Pr: float) -> float:
    """Compute Nusselt number: Nu = α·Re^n·Pr^m"""
    (alpha, n, m) = params
    Nu = alpha * exp(log(Re) * n) * exp(log(Pr) * m)
    return Nu


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


def hcorrelation(
    params: tuple,
    Tw: float,
    Pw: float,
    dPw: float,
    U: float,
    Dh: float,
    L: float,
    friction: str = "Constant",
    pextra: float = 1,
    model: str = "Montgomery",
    rugosity: float = 0.012e-3,
) -> float:
    """
    Compute heat exchange coefficient [W/m²/K].

    Formula: h = α · Re^n · Pr^m · k / Dh

    Args:
        params:   Correlation coefficients (α, n, m)
        Tw:       Water temperature [K]
        Pw:       Water pressure [bar]
        dPw:      Pressure drop [bar]
        U:        Initial velocity guess [m/s]
        Dh:       Hydraulic diameter [m]
        L:        Channel length [m]
        friction: Friction model name
        pextra:   Additional pressure loss coefficient
        model:    Correlation name (for logging)
        rugosity: Surface roughness [m]

    Returns:
        Heat transfer coefficient [W/m²/K]
    """
    (alpha, n, m) = params
    state = steam(Tw, Pw)
    nU, _ = Uw(
        state,
        dPw,
        Dh,
        L,
        friction,
        Pextra=pextra,
        fguess=0.055,
        uguess=U,
        rugosity=rugosity,
    )

    Re = Reynolds(state, nU, Dh, L)
    Pr = Prandtl(state)

    h = alpha * exp(log(Re) * n) * exp(log(Pr) * m) * state.thermal_conductivity / Dh
    return h


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
):
    """
    Compute heat transfer coefficient for the given correlation model [W/m²/K].

    Args:
        Dh:       Hydraulic diameter [m]
        L:        Channel length [m]
        U:        Initial velocity guess [m/s]
        Tw:       Water temperature [K]
        Pw:       Water pressure [bar]
        dPw:      Pressure drop [bar]
        model:    Correlation name (Montgomery, Dittus, Colburn, Silverberg)
        friction: Friction model name
        fuzzy:    Empirical correction factor
        pextra:   Additional pressure loss coefficient

    Returns:
        Heat transfer coefficient [W/m²/K]
    """
    correlation = {
        "Montgomery": Montgomery,
        "Dittus": Dittus,
        "Colburn": Colburn,
        "Silverberg": Silverberg,
    }

    return correlation[model](Tw, Pw, dPw, U, Dh, L, friction, fuzzy, pextra)


def getTout(T: List[float], VolMass: List[float], SpecHeat: List[float], Q: List[float]) -> float:
    """
    Compute energy-weighted mixed outlet temperature.

    Formula: T_out = Σ(Tᵢ·ρᵢ·cpᵢ·Qᵢ) / Σ(ρᵢ·cpᵢ·Qᵢ)

    Args:
        T:        Outlet temperatures per channel [K]
        VolMass:  Water densities per channel [kg/m³]
        SpecHeat: Specific heats per channel [J/kg/K]
        Q:        Volumetric flow rates per channel [m³/s]

    Returns:
        Mixed outlet temperature [K]
    """
    Tout = 0
    rhoCpQ = 0
    for Ti, RHOi, CPi, Qi in zip(T, VolMass, SpecHeat, Q):
        Tout += Ti * RHOi * CPi * Qi
        rhoCpQ += RHOi * CPi * Qi

    Tout /= rhoCpQ
    return Tout
