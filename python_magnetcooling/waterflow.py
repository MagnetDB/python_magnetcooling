"""
Water flow and pump characteristics for magnet cooling systems.

This module handles:
- Pump characteristic curves (flow vs speed, pressure vs speed)
- Flow rate calculations
- Pressure drop calculations
- Velocity calculations based on operating conditions
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
import numpy as np
from warnings import simplefilter
from pint import UnitRegistry, Quantity

# Ignore pint warnings
simplefilter("ignore")
Quantity([])

# Pint configuration
ureg = UnitRegistry()
ureg.default_system = "SI"
ureg.autoconvert_offset_to_baseunit = True

# Import hysteresis model from hysteresis module
from .hysteresis import multi_level_hysteresis as _multi_level_hysteresis


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
        hysteresis_thresholds: List of (ascending, descending) power threshold tuples [MW]
        hysteresis_low_values: Flow rates for low state at each level [m³/h]
        hysteresis_high_values: Flow rates for high state at each level [m³/h]
    
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
    hysteresis_thresholds: List[tuple] = field(default_factory=list)  # [(asc, desc), ...] in MW
    hysteresis_low_values: List[float] = field(default_factory=list)  # m³/h
    hysteresis_high_values: List[float] = field(default_factory=list)  # m³/h
    
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
        
        # Load hysteresis parameters if present
        hysteresis_thresholds = []
        hysteresis_low_values = []
        hysteresis_high_values = []
        
        if "hysteresis" in params:
            hyst = params["hysteresis"]
            # Convert thresholds to list of tuples if they're lists
            thresholds_raw = hyst.get("thresholds", [])
            if thresholds_raw and isinstance(thresholds_raw[0], (list, tuple)):
                hysteresis_thresholds = [tuple(t) for t in thresholds_raw]
            else:
                hysteresis_thresholds = thresholds_raw
            hysteresis_low_values = hyst.get("low_values", [])
            hysteresis_high_values = hyst.get("high_values", [])
        
        return cls(
            pump_speed_min=params["Vp0"]["value"],
            pump_speed_max=params["Vpmax"]["value"],
            flow_min=params["F0"]["value"],
            flow_max=params["Fmax"]["value"],
            pressure_max=params["Pmax"]["value"],
            pressure_min=params["Pmin"]["value"],
            pressure_back=params.get("BP", {"value": 4})["value"],
            current_max=params["Imax"]["value"],
            hysteresis_thresholds=hysteresis_thresholds,
            hysteresis_low_values=hysteresis_low_values,
            hysteresis_high_values=hysteresis_high_values
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
    
    def debitbrut(self, power: float) -> float:
        """
        Compute secondary cooling loop flow rate as function of power using hysteresis model.
        
        This method uses a multi-level hysteresis model to account for the fact that
        flow rate control depends not only on current power but also on the direction
        of power change (increasing vs decreasing).
        
        Note: 'debitbrut' is the original French term for secondary loop flow rate.
        In new CSV files, prefer the column name 'flow_secondary' for clarity.
        
        Based on the hysteresis model from examples/hysteresis.py
        
        Args:
            power: Magnet power [MW] (scalar or array)
            
        Returns:
            Secondary cooling loop flow rate [m³/h]
            
        Raises:
            ValueError: If hysteresis parameters are not configured
            
        Note:
            Hysteresis parameters must be set either:
            - Via from_file() loading a JSON with "hysteresis" section
            - By directly setting hysteresis_thresholds (as list of tuples),
              hysteresis_low_values, and hysteresis_high_values attributes
              
        Example:
            >>> flow = WaterFlow()
            >>> # Each threshold is (ascending, descending) tuple
            >>> flow.hysteresis_thresholds = [(3, 2), (8, 6), (12, 10)]
            >>> flow.hysteresis_low_values = [100, 200, 300, 400]
            >>> flow.hysteresis_high_values = [100, 250, 350, 450]
            >>> power_sequence = np.array([0, 5, 10, 15, 10, 5, 0])
            >>> flow_rates = flow.debitbrut(power_sequence)
        """
        if not self.hysteresis_thresholds:
            raise ValueError(
                "Hysteresis parameters not configured. "
                "Set hysteresis_thresholds (as list of (asc, desc) tuples), "
                "hysteresis_low_values, and hysteresis_high_values before calling debitbrut()."
            )
        
        if len(self.hysteresis_low_values) != len(self.hysteresis_thresholds):
            raise ValueError(
                f"hysteresis_low_values must have {len(self.hysteresis_thresholds)} "
                f"elements (same as number of thresholds), got {len(self.hysteresis_low_values)}"
            )
        
        if len(self.hysteresis_high_values) != len(self.hysteresis_thresholds):
            raise ValueError(
                f"hysteresis_high_values must have {len(self.hysteresis_thresholds)} "
                f"elements (same as number of thresholds), got {len(self.hysteresis_high_values)}"
            )
        
        # Convert scalar to array for processing
        power_array = np.atleast_1d(power)
        
        # Apply hysteresis model
        flow_rates = _multi_level_hysteresis(
            power_array,
            self.hysteresis_thresholds,
            self.hysteresis_low_values,
            self.hysteresis_high_values
        )
        
        # Return scalar if input was scalar
        if np.isscalar(power):
            return float(flow_rates[0])
        return flow_rates
    
    def to_dict(self) -> dict:
        """Export parameters as dictionary"""
        result = {
            "Vp0": {"value": self.pump_speed_min, "unit": "rpm"},
            "Vpmax": {"value": self.pump_speed_max, "unit": "rpm"},
            "F0": {"value": self.flow_min, "unit": "l/s"},
            "Fmax": {"value": self.flow_max, "unit": "l/s"},
            "Pmax": {"value": self.pressure_max, "unit": "bar"},
            "Pmin": {"value": self.pressure_min, "unit": "bar"},
            "BP": {"value": self.pressure_back, "unit": "bar"},
            "Imax": {"value": self.current_max, "unit": "A"},
        }
        
        # Add hysteresis parameters if configured
        if self.hysteresis_thresholds:
            result["hysteresis"] = {
                "thresholds": self.hysteresis_thresholds,
                "low_values": self.hysteresis_low_values,
                "high_values": self.hysteresis_high_values,
                "unit_thresholds": "MW",
                "unit_values": "m³/h"
            }
        
        return result
    
    def to_file(self, filename: str) -> None:
        """Save parameters to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
