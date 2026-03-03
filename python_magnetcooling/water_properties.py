"""
Water and steam properties using IAPWS-IF97 standard.

This module provides a clean interface to water/steam properties
needed for thermal-hydraulic calculations.
"""

from typing import NamedTuple
from iapws import IAPWS97
from .exceptions import WaterPropertiesError


class WaterState(NamedTuple):
    """Water/steam thermodynamic state"""
    temperature: float  # K
    pressure: float  # bar
    density: float  # kg/m³
    specific_heat: float  # J/kg/K
    thermal_conductivity: float  # W/m/K
    dynamic_viscosity: float  # Pa·s
    prandtl: float  # dimensionless


class WaterProperties:
    """
    Calculate water/steam properties using IAPWS-IF97
    
    Examples:
        >>> water = WaterProperties()
        >>> state = water.get_state(temperature=300.0, pressure=10.0)
        >>> print(f"Density: {state.density:.2f} kg/m³")
        >>> print(f"Viscosity: {state.dynamic_viscosity:.6f} Pa·s")
    """
    
    @staticmethod
    def get_state(temperature: float, pressure: float) -> WaterState:
        """
        Get water properties at given conditions
        
        Args:
            temperature: Temperature [K]
            pressure: Pressure [bar]
            
        Returns:
            Complete water state
            
        Raises:
            WaterPropertiesError: If conditions are outside valid range
        """
        try:
            pressure_mpa = pressure * 0.1
            steam = IAPWS97(P=pressure_mpa, T=temperature)
            
            if not steam.status:
                raise WaterPropertiesError(
                    f"Invalid water properties state: T={temperature} K, P={pressure} bar"
                )
            
            # Calculate Prandtl number
            prandtl = steam.mu * steam.cp * 1e3 / steam.k
            
            return WaterState(
                temperature=temperature,
                pressure=pressure,
                density=steam.rho,
                specific_heat=steam.cp * 1e3,  # Convert kJ/kg/K to J/kg/K
                thermal_conductivity=steam.k,
                dynamic_viscosity=steam.mu,
                prandtl=prandtl
            )
        except Exception as e:
            raise WaterPropertiesError(
                f"Error calculating water properties: {e}"
            ) from e
    
    @staticmethod
    def compute_temperature_rise(
        flow_rate: float,
        power: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Compute temperature rise: ΔT = Q / (ρ·cp·V̇)
        
        Args:
            flow_rate: Volumetric flow rate [m³/s]
            power: Heat power [W]
            temperature: Water temperature [K]
            pressure: Water pressure [bar]
            
        Returns:
            Temperature rise [K]
        """
        state = WaterProperties.get_state(temperature, pressure)
        return power / (state.density * state.specific_heat * flow_rate)
    
    @staticmethod
    def compute_reynolds(
        velocity: float,
        hydraulic_diameter: float,
        temperature: float,
        pressure: float
    ) -> float:
        """
        Compute Reynolds number: Re = ρ·U·Dh/μ
        
        Args:
            velocity: Flow velocity [m/s]
            hydraulic_diameter: Hydraulic diameter [m]
            temperature: Water temperature [K]
            pressure: Water pressure [bar]
            
        Returns:
            Reynolds number [dimensionless]
        """
        state = WaterProperties.get_state(temperature, pressure)
        return state.density * velocity * hydraulic_diameter / state.dynamic_viscosity

# add helpers
def get_rho(pbar: float, celsius: float) -> float:
    """Water density [kg/m³]. Args: pressure (bar), temperature (°C)."""
    return WaterProperties.get_state(temperature=celsius + 273.15, pressure=pbar).density


def get_cp(pbar: float, celsius: float) -> float:
    """Water specific heat [J/kg/K]. Args: pressure (bar), temperature (°C)."""
    return WaterProperties.get_state(temperature=celsius + 273.15, pressure=pbar).specific_heat
