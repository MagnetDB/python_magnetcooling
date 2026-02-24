"""
Heat transfer correlations for internal forced convection.

Implements various Nusselt number correlations:
- Montgomery: Semi-empirical for high heat flux magnets
- Dittus-Boelter: Classic turbulent correlation
- Colburn: Modified Dittus-Boelter
- Silverberg: High heat flux correlation
"""

from abc import ABC, abstractmethod
from typing import Dict, Type
from math import exp, log
from .water_properties import WaterProperties
from .exceptions import CorrelationError

# 0 °C expressed in Kelvin.  Used in correlations that were originally
# defined for temperatures in Celsius (e.g. Montgomery).
_CELSIUS_ZERO_K: float = 273.15


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
        assert temperature > 0, f"Temperature must be positive [K], got {temperature}"
        assert velocity > 0, f"Velocity must be positive [m/s], got {velocity}"
        assert hydraulic_diameter > 0, f"Hydraulic diameter must be positive [m], got {hydraulic_diameter}"

        # Convert temperature to Celsius for the correlation
        temp_celsius = temperature - _CELSIUS_ZERO_K

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
        assert temperature > 0, f"Temperature must be positive [K], got {temperature}"
        assert velocity > 0, f"Velocity must be positive [m/s], got {velocity}"
        assert hydraulic_diameter > 0, f"Hydraulic diameter must be positive [m], got {hydraulic_diameter}"

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
        assert temperature > 0, f"Temperature must be positive [K], got {temperature}"
        assert velocity > 0, f"Velocity must be positive [m/s], got {velocity}"
        assert hydraulic_diameter > 0, f"Hydraulic diameter must be positive [m], got {hydraulic_diameter}"

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
        assert temperature > 0, f"Temperature must be positive [K], got {temperature}"
        assert velocity > 0, f"Velocity must be positive [m/s], got {velocity}"
        assert hydraulic_diameter > 0, f"Hydraulic diameter must be positive [m], got {hydraulic_diameter}"

        state = WaterProperties.get_state(temperature, pressure)
        reynolds = WaterProperties.compute_reynolds(
            velocity, hydraulic_diameter, temperature, pressure
        )
        
        nusselt = self.compute_nusselt(reynolds, state.prandtl, 0.015, 0.85, 0.3)
        h = self.fuzzy_factor * nusselt * state.thermal_conductivity / hydraulic_diameter
        
        return h


# Registry of available correlations
_CORRELATIONS: Dict[str, Type[HeatCorrelation]] = {
    "Montgomery": MontgomeryCorrelation,
    "Dittus": DittusBoelterCorrelation,
    "Colburn": ColburnCorrelation,
    "Silverberg": SilverbergCorrelation,
}


def get_correlation(name: str, fuzzy_factor: float = 1.0) -> HeatCorrelation:
    """
    Get heat transfer correlation by name
    
    Args:
        name: Correlation name (Montgomery, Dittus, Colburn, Silverberg)
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
