"""
Python Magnet Cooling
=====================

Thermal-hydraulic calculations for water-cooled high-field magnets.

Main Components
--------------
- thermal_hydraulics: Main solver for cooling calculations
- waterflow: Pump characteristics and flow rate calculations
- correlations: Heat transfer correlations (Montgomery, Dittus-Boelter, etc.)
- water_properties: Water/steam properties (IAPWS-IF97)
- friction: Friction factor models

Quick Start
-----------
>>> from python_magnetcooling import compute_single_channel
>>> result = compute_single_channel(
...     hydraulic_diameter=0.008,
...     cross_section=5e-5,
...     length=0.5,
...     power=50000,
...     temp_inlet=290.0,
...     pressure_inlet=15.0,
...     pressure_drop=5.0
... )
>>> print(f"Outlet temp: {result.temp_outlet:.2f} K")
"""

__author__ = "Christophe Trophime"
__email__ = "christophe.trophime@lncmi.cnrs.fr"

# Version is read from package metadata (defined in pyproject.toml)
# This ensures a single source of truth for the version number
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Fallback for Python < 3.8 (though we require 3.11+)
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("python-magnetcooling")
except PackageNotFoundError:
    # Package not installed (e.g., running from source without install)
    # This is expected during development before running `pip install -e .`
    __version__ = "0.0.0+unknown"

from .thermohydraulics import (
    ThermalHydraulicCalculator,
    ThermalHydraulicInput,
    ThermalHydraulicOutput,
    compute_single_channel,
)
from .channel import (
    ChannelGeometry,
    ChannelInput,
    ChannelOutput,
    AxialDiscretization,
)
from .waterflow import WaterFlow
from .waterflow_factory import (
    from_flow_params,
    from_database_record,
    from_fitted_data,
    create_default as create_default_waterflow,
)
from .correlations import HeatCorrelation, available_correlations
from .friction import FrictionModel, available_friction_models
from .water_properties import WaterProperties, get_rho, get_cp

__all__ = [
    "__version__",
    # Main calculator
    "ThermalHydraulicCalculator",
    "ThermalHydraulicInput",
    "ThermalHydraulicOutput",
    "compute_single_channel",
    # Channel definitions
    "ChannelGeometry",
    "ChannelInput",
    "ChannelOutput",
    "AxialDiscretization",
    # Components
    "WaterFlow",
    "WaterProperties",
    "HeatCorrelation",
    "FrictionModel",
    # WaterFlow factory functions
    "from_flow_params",
    "from_database_record",
    "from_fitted_data",
    "create_default_waterflow",
    # Utilities
    "available_correlations",
    "available_friction_models",
]

