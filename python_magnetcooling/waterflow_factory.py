"""
Factory functions for creating WaterFlow objects from various data sources.

This module provides utilities to construct WaterFlow instances from:
- Database records
- Flow parameter dictionaries
- Fitted experimental data
"""

from typing import Dict, Any, Optional
from python_magnetcooling.waterflow import WaterFlow


def from_flow_params(params: Dict[str, Any]) -> WaterFlow:
    """
    Create WaterFlow object from flow parameters dictionary.
    
    This is the primary factory method for creating WaterFlow instances from
    database records or fitted experimental data. The dictionary structure
    matches the format used in database exports and fitting procedures.
    
    Args:
        params: Flow parameters dictionary with the following structure:
            {
                "Vp0": {"value": float, "unit": "rpm"},      # Min pump speed
                "Vpmax": {"value": float, "unit": "rpm"},    # Max pump speed
                "F0": {"value": float, "unit": "l/s"},       # Min flow rate
                "Fmax": {"value": float, "unit": "l/s"},     # Max flow rate
                "Pmax": {"value": float, "unit": "bar"},     # Max pressure
                "Pmin": {"value": float, "unit": "bar"},     # Min pressure
                "BP": {"value": float, "unit": "bar"},       # Back pressure
                "Imax": {"value": float, "unit": "A"}        # Max current
            }
            
            Note: "BP" and "Pout" are equivalent (back pressure)
            
    Returns:
        WaterFlow instance configured with the provided parameters
        
    Examples:
        >>> params = {
        ...     "Vp0": {"value": 1000, "unit": "rpm"},
        ...     "Vpmax": {"value": 2840, "unit": "rpm"},
        ...     "F0": {"value": 0, "unit": "l/s"},
        ...     "Fmax": {"value": 140, "unit": "l/s"},
        ...     "Pmax": {"value": 22, "unit": "bar"},
        ...     "Pmin": {"value": 4, "unit": "bar"},
        ...     "BP": {"value": 4, "unit": "bar"},
        ...     "Imax": {"value": 28000, "unit": "A"}
        ... }
        >>> flow = from_flow_params(params)
        >>> print(f"Max flow: {flow.flow_max} l/s")
        
    Raises:
        KeyError: If required parameters are missing
        ValueError: If parameter values are invalid
    """
    # Handle both "BP" and "Pout" keys for back pressure
    back_pressure_key = "BP" if "BP" in params else "Pout"
    
    return WaterFlow(
        pump_speed_min=params["Vp0"]["value"],
        pump_speed_max=params["Vpmax"]["value"],
        flow_min=params["F0"]["value"],
        flow_max=params["Fmax"]["value"],
        pressure_max=params["Pmax"]["value"],
        pressure_min=params["Pmin"]["value"],
        pressure_back=params.get(back_pressure_key, {"value": 4})["value"],
        current_max=params["Imax"]["value"]
    )


def from_database_record(
    record: Dict[str, Any],
    key_mapping: Optional[Dict[str, str]] = None
) -> WaterFlow:
    """
    Create WaterFlow object from a database record.
    
    Provides flexibility for different database schemas by allowing custom
    key mappings between database fields and WaterFlow parameters.
    
    Args:
        record: Database record dictionary
        key_mapping: Optional mapping of database keys to flow parameter keys.
            If None, assumes the record uses standard flow_params format.
            
    Returns:
        WaterFlow instance
        
    Examples:
        >>> record = {
        ...     "min_pump_rpm": 1000,
        ...     "max_pump_rpm": 2840,
        ...     "min_flow_rate": 0,
        ...     "max_flow_rate": 140,
        ...     "max_pressure": 22,
        ...     "min_pressure": 4,
        ...     "back_pressure": 4,
        ...     "max_current": 28000
        ... }
        >>> mapping = {
        ...     "Vp0": "min_pump_rpm",
        ...     "Vpmax": "max_pump_rpm",
        ...     "F0": "min_flow_rate",
        ...     "Fmax": "max_flow_rate",
        ...     "Pmax": "max_pressure",
        ...     "Pmin": "min_pressure",
        ...     "BP": "back_pressure",
        ...     "Imax": "max_current"
        ... }
        >>> flow = from_database_record(record, mapping)
    """
    if key_mapping:
        # Convert database record to flow_params format using mapping
        params = {}
        for flow_key, db_key in key_mapping.items():
            if db_key in record:
                # If record value is dict with "value" key, use it directly
                if isinstance(record[db_key], dict) and "value" in record[db_key]:
                    params[flow_key] = record[db_key]
                else:
                    # Otherwise wrap the value
                    params[flow_key] = {"value": record[db_key]}
        return from_flow_params(params)
    else:
        # Assume record is already in flow_params format
        return from_flow_params(record)


def create_default() -> WaterFlow:
    """
    Create WaterFlow object with default parameter values.
    
    These defaults are typical for magnet cooling systems but should be
    adjusted based on actual system specifications.
    
    Returns:
        WaterFlow instance with default parameters
        
    Examples:
        >>> flow = create_default()
        >>> print(f"Default max current: {flow.current_max} A")
    """
    return WaterFlow()


def from_fitted_data(
    pump_speed_fit: tuple,
    flow_rate_fit: tuple,
    pressure_fit: tuple,
    back_pressure: float,
    max_current: float
) -> WaterFlow:
    """
    Create WaterFlow from fitted curve parameters.
    
    This is useful when flow parameters are determined by curve fitting
    experimental data, as done in the compute() function of flow_params.py.
    
    Args:
        pump_speed_fit: (Vpmax, Vp0) tuple from pump speed fit
        flow_rate_fit: (F0, Fmax) tuple from flow rate fit
        pressure_fit: (Pmin, Pmax) tuple from pressure fit
        back_pressure: Back pressure value [bar]
        max_current: Maximum operating current [A]
        
    Returns:
        WaterFlow instance
        
    Examples:
        >>> # From fitting: Vp = Vpmax*(I/Imax)^2 + Vp0
        >>> pump_fit = (2840, 1000)  # (Vpmax, Vp0)
        >>> flow_fit = (0, 140)      # (F0, Fmax)
        >>> press_fit = (4, 22)      # (Pmin, Pmax)
        >>> flow = from_fitted_data(pump_fit, flow_fit, press_fit, 4.0, 28000)
    """
    return WaterFlow(
        pump_speed_max=pump_speed_fit[0],
        pump_speed_min=pump_speed_fit[1],
        flow_min=flow_rate_fit[0],
        flow_max=flow_rate_fit[1],
        pressure_min=pressure_fit[0],
        pressure_max=pressure_fit[1],
        pressure_back=back_pressure,
        current_max=max_current
    )
