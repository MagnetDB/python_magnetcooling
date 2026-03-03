"""
Heat transfer correlations for internal forced convection.

Implements various Nusselt number correlations:
- Montgomery: Semi-empirical for high heat flux magnets
- Dittus-Boelter: Classic turbulent correlation
- Colburn: Modified Dittus-Boelter
- Silverberg: High heat flux correlation
- Gnielinski: Improved turbulent correlation (more accurate near transition)
"""

from abc import ABC, abstractmethod
from typing import Dict, Type
from math import exp, log
from .water_properties import WaterProperties
from .exceptions import CorrelationError


class HeatCorrelation(ABC):
    """Base class for heat transfer correlations"""
    
    def __init__(self, fuzzy_factor: float = 1.0):
        """
        Initialize correlation
        
        Args:
            fuzzy_factor: Empirical correction factor (default: 1.0)
        """
        self.fuzzy_factor = fuzzy_factor
    
    @abstractmethod
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """
        Compute heat transfer coefficient
        
        Args:
            temperature: Water temperature [K]
            pressure: Water pressure [bar]
            velocity: Flow velocity [m/s]
            hydraulic_diameter: Hydraulic diameter [m]
            length: Channel length [m]
            
        Returns:
            Heat transfer coefficient [W/m²/K]
        """
        pass
    
    @staticmethod
    def compute_nusselt(
        reynolds: float,
        prandtl: float,
        alpha: float,
        n: float,
        m: float
    ) -> float:
        """
        Generic Nusselt correlation: Nu = α·Re^n·Pr^m
        
        Args:
            reynolds: Reynolds number
            prandtl: Prandtl number
            alpha: Correlation coefficient
            n: Reynolds exponent
            m: Prandtl exponent
            
        Returns:
            Nusselt number
        """
        return alpha * exp(log(reynolds) * n) * exp(log(prandtl) * m)


class MontgomeryCorrelation(HeatCorrelation):
    """
    Montgomery correlation for high heat flux applications
    
    Reference: Montgomery, D.B. "Solenoid Magnet Design" (1969)
    Formula: h = fuzzy · 1426.404 · (1 + 0.015·(T-273)) · U^0.8 / Dh^0.2
    
    Note: Original formula uses T in Celsius and Dh in centimeters
    """
    
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """Compute heat transfer coefficient using Montgomery correlation"""
        
        # Convert temperature to Celsius for the correlation
        temp_celsius = temperature - 273.15
        
        # Convert Dh to cm for the correlation
        dh_cm = hydraulic_diameter * 100.0
        
        h = (
            self.fuzzy_factor
            * 1426.404
            * (1.0 + 0.015 * temp_celsius)
            * exp(log(velocity) * 0.8)
            / exp(log(dh_cm) * 0.2)
        )
        
        return h


class DittusBoelterCorrelation(HeatCorrelation):
    """
    Dittus-Boelter correlation for turbulent flow
    
    Nu = 0.023 · Re^0.8 · Pr^0.4 (heating)
    Valid for: Re > 10000, 0.7 < Pr < 160, L/D > 10
    """
    
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """Compute heat transfer coefficient using Dittus-Boelter"""
        
        state = WaterProperties.get_state(temperature, pressure)
        reynolds = WaterProperties.compute_reynolds(
            velocity, hydraulic_diameter, temperature, pressure
        )
        
        if reynolds < 2300:
            raise CorrelationError(
                f"Dittus-Boelter not valid for laminar flow (Re={reynolds:.0f} < 2300)"
            )
        
        nusselt = self.compute_nusselt(reynolds, state.prandtl, 0.023, 0.8, 0.4)
        h = self.fuzzy_factor * nusselt * state.thermal_conductivity / hydraulic_diameter
        
        return h


class ColburnCorrelation(HeatCorrelation):
    """
    Colburn correlation (modified Dittus-Boelter)
    
    Nu = 0.023 · Re^0.8 · Pr^0.3
    More conservative than Dittus-Boelter
    """
    
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """Compute heat transfer coefficient using Colburn"""
        
        state = WaterProperties.get_state(temperature, pressure)
        reynolds = WaterProperties.compute_reynolds(
            velocity, hydraulic_diameter, temperature, pressure
        )
        
        nusselt = self.compute_nusselt(reynolds, state.prandtl, 0.023, 0.8, 0.3)
        h = self.fuzzy_factor * nusselt * state.thermal_conductivity / hydraulic_diameter
        
        return h


class SilverbergCorrelation(HeatCorrelation):
    """
    Silverberg correlation for high heat flux
    
    Nu = 0.015 · Re^0.85 · Pr^0.3
    Developed for high heat flux applications
    """
    
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """Compute heat transfer coefficient using Silverberg"""
        
        state = WaterProperties.get_state(temperature, pressure)
        reynolds = WaterProperties.compute_reynolds(
            velocity, hydraulic_diameter, temperature, pressure
        )
        
        nusselt = self.compute_nusselt(reynolds, state.prandtl, 0.015, 0.85, 0.3)
        h = self.fuzzy_factor * nusselt * state.thermal_conductivity / hydraulic_diameter
        
        return h


class GnielinskiCorrelation(HeatCorrelation):
    """
    Gnielinski correlation for turbulent flow in smooth pipes
    
    Nu = (f/8)(Re - 1000)Pr / [1 + 12.7(f/8)^0.5(Pr^(2/3) - 1)]
    
    where f is the friction factor (Petukhov: f = (0.79·ln(Re) - 1.64)^(-2))
    
    Reference: Gnielinski, V. (1976). "New equations for heat and mass transfer 
    in turbulent pipe and channel flow." Int. Chem. Eng., 16(2), 359-368.
    
    Valid for:
    - 0.5 < Pr < 2000
    - 3000 < Re < 5×10^6
    - Smooth pipes
    
    More accurate than Dittus-Boelter, especially near transition region.
    """
    
    def compute(
        self,
        temperature: float,
        pressure: float,
        velocity: float,
        hydraulic_diameter: float,
        length: float
    ) -> float:
        """Compute heat transfer coefficient using Gnielinski correlation"""
        
        state = WaterProperties.get_state(temperature, pressure)
        reynolds = WaterProperties.compute_reynolds(
            velocity, hydraulic_diameter, temperature, pressure
        )
        
        if reynolds < 3000:
            raise CorrelationError(
                f"Gnielinski not valid for Re={reynolds:.0f} < 3000 (use laminar correlation)"
            )
        
        if reynolds > 5e6:
            raise CorrelationError(
                f"Gnielinski not recommended for Re={reynolds:.0f} > 5×10^6"
            )
        
        prandtl = state.prandtl
        
        if prandtl < 0.5 or prandtl > 2000:
            raise CorrelationError(
                f"Gnielinski not valid for Pr={prandtl:.2f} outside range [0.5, 2000]"
            )
        
        # Petukhov friction factor for smooth pipes
        f = (0.79 * log(reynolds) - 1.64) ** (-2)
        
        # Gnielinski Nusselt number
        numerator = (f / 8.0) * (reynolds - 1000) * prandtl
        denominator = 1.0 + 12.7 * (f / 8.0) ** 0.5 * (prandtl ** (2.0/3.0) - 1.0)
        nusselt = numerator / denominator
        
        h = self.fuzzy_factor * nusselt * state.thermal_conductivity / hydraulic_diameter
        
        return h


# Registry of available correlations
_CORRELATIONS: Dict[str, Type[HeatCorrelation]] = {
    "Montgomery": MontgomeryCorrelation,
    "Dittus": DittusBoelterCorrelation,
    "Colburn": ColburnCorrelation,
    "Silverberg": SilverbergCorrelation,
    "Gnielinski": GnielinskiCorrelation,
}


def get_correlation(name: str, fuzzy_factor: float = 1.0) -> HeatCorrelation:
    """
    Get heat transfer correlation by name
    
    Args:
        name: Correlation name (Montgomery, Dittus, Colburn, Silverberg, Gnielinski)
        fuzzy_factor: Empirical correction factor
        
    Returns:
        Correlation instance
        
    Raises:
        CorrelationError: If correlation name is unknown
    """
    if name not in _CORRELATIONS:
        raise CorrelationError(
            f"Unknown correlation '{name}'. Available: {list(_CORRELATIONS.keys())}"
        )
    
    return _CORRELATIONS[name](fuzzy_factor=fuzzy_factor)


def available_correlations() -> list[str]:
    """Get list of available heat transfer correlations"""
    return list(_CORRELATIONS.keys())
