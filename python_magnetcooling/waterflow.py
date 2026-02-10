"""
Water flow and pump characteristics for magnet cooling systems.

This module handles:
- Pump characteristic curves (flow vs speed, pressure vs speed)
- Flow rate calculations
- Pressure drop calculations
- Velocity calculations based on operating conditions
"""

from dataclasses import dataclass
from typing import Optional
import json
from warnings import simplefilter
from pint import UnitRegistry, Quantity

# Ignore pint warnings
simplefilter("ignore")
Quantity([])

# Pint configuration
ureg = UnitRegistry()
ureg.default_system = "SI"
ureg.autoconvert_offset_to_baseunit = True


@dataclass
class WaterFlow:
    """
    Water flow system characteristics
    
    Represents pump curves and hydraulic system parameters for
    water-cooled magnet systems.
    
    Attributes:
        pump_speed_min: Minimum pump speed [rpm]
        pump_speed_max: Maximum pump speed [rpm]
        flow_min: Flow rate at minimum speed [l/s]
        flow_max: Flow rate at maximum speed [l/s]
        pressure_max: Maximum pressure [bar]
        pressure_min: Minimum pressure [bar]
        pressure_back: Back pressure [bar]
        current_max: Maximum operating current [A]
    
    Examples:
        >>> flow = WaterFlow(
        ...     pump_speed_min=1000,
        ...     pump_speed_max=2840,
        ...     flow_min=0,
        ...     flow_max=140,
        ...     pressure_max=22,
        ...     pressure_min=4,
        ...     pressure_back=4,
        ...     current_max=28000
        ... )
        >>> p = flow.pressure(20000)
        >>> print(f"Pressure at 20kA: {p:.2f} bar")
    """
    
    pump_speed_min: float = 1000  # rpm
    pump_speed_max: float = 2840  # rpm
    flow_min: float = 0  # l/s
    flow_max: float = 140  # l/s
    pressure_max: float = 22  # bar
    pressure_min: float = 4  # bar
    pressure_back: float = 4  # bar
    current_max: float = 28000  # A
    
    @classmethod
    def from_file(cls, filename: str) -> "WaterFlow":
        """
        Load flow parameters from JSON file
        
        Args:
            filename: Path to JSON file with flow parameters
            
        Returns:
            WaterFlow instance
            
        Example JSON format:
        {
            "Vp0": {"value": 1000, "unit": "rpm"},
            "Vpmax": {"value": 2840, "unit": "rpm"},
            "F0": {"value": 0, "unit": "l/s"},
            "Fmax": {"value": 140, "unit": "l/s"},
            "Pmax": {"value": 22, "unit": "bar"},
            "Pmin": {"value": 4, "unit": "bar"},
            "BP": {"value": 4, "unit": "bar"},
            "Imax": {"value": 28000, "unit": "A"}
        }
        """
        with open(filename, "r") as f:
            params = json.load(f)
        
        return cls(
            pump_speed_min=params["Vp0"]["value"],
            pump_speed_max=params["Vpmax"]["value"],
            flow_min=params["F0"]["value"],
            flow_max=params["Fmax"]["value"],
            pressure_max=params["Pmax"]["value"],
            pressure_min=params["Pmin"]["value"],
            pressure_back=params.get("BP", {"value": 4})["value"],
            current_max=params["Imax"]["value"]
        )
    
    def pump_speed(self, current: float) -> float:
        """
        Compute pump speed as function of current
        
        Assumes quadratic relationship: Vp = Vp_max·(I/I_max)² + Vp_min
        
        Args:
            current: Operating current [A]
            
        Returns:
            Pump speed [rpm]
        """
        if current >= self.current_max:
            return self.pump_speed_max + self.pump_speed_min
        
        return (
            self.pump_speed_max * (current / self.current_max) ** 2
            + self.pump_speed_min
        )
    
    def flow_rate(self, current: float) -> float:
        """
        Compute flow rate as function of current
        
        Args:
            current: Operating current [A]
            
        Returns:
            Flow rate [m³/s]
        """
        # Convert l/s to m³/s
        units = [
            ureg.liter / ureg.second,
            ureg.meter ** 3 / ureg.second,
        ]
        
        F0 = Quantity(self.flow_min, units[0]).to(units[1]).magnitude
        Fmax = Quantity(self.flow_max, units[0]).to(units[1]).magnitude
        
        if current >= self.current_max:
            return F0 + Fmax
        
        Vp_total = self.pump_speed_max + self.pump_speed_min
        return F0 + Fmax * self.pump_speed(current) / Vp_total
    
    def pressure(self, current: float) -> float:
        """
        Compute system pressure as function of current
        
        Assumes quadratic relationship with pump speed
        
        Args:
            current: Operating current [A]
            
        Returns:
            Pressure [bar]
        """
        if current >= self.current_max:
            return self.pressure_min + self.pressure_max
        
        Vp_total = self.pump_speed_max + self.pump_speed_min
        return (
            self.pressure_min
            + self.pressure_max * (self.pump_speed(current) / Vp_total) ** 2
        )
    
    def pressure_drop(self, current: float) -> float:
        """
        Compute pressure drop (pressure minus back pressure)
        
        Args:
            current: Operating current [A]
            
        Returns:
            Pressure drop [bar]
        """
        return self.pressure(current) - self.pressure_back
    
    def velocity(self, current: float, cross_section: float) -> float:
        """
        Compute mean velocity from flow rate and cross-section
        
        Args:
            current: Operating current [A]
            cross_section: Total cross-sectional area [m²]
            
        Returns:
            Mean velocity [m/s]
        """
        if cross_section <= 0:
            raise ValueError("Cross-section must be positive")
        
        return self.flow_rate(current) / cross_section
    
    def to_dict(self) -> dict:
        """Export parameters as dictionary"""
        return {
            "Vp0": {"value": self.pump_speed_min, "unit": "rpm"},
            "Vpmax": {"value": self.pump_speed_max, "unit": "rpm"},
            "F0": {"value": self.flow_min, "unit": "l/s"},
            "Fmax": {"value": self.flow_max, "unit": "l/s"},
            "Pmax": {"value": self.pressure_max, "unit": "bar"},
            "Pmin": {"value": self.pressure_min, "unit": "bar"},
            "BP": {"value": self.pressure_back, "unit": "bar"},
            "Imax": {"value": self.current_max, "unit": "A"},
        }
    
    def to_file(self, filename: str) -> None:
        """Save parameters to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
