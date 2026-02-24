"""
Channel geometry and data structures for thermal-hydraulic calculations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CoolingLevel(str, Enum):
    """
    Cooling model hierarchy for water-cooled magnets.

    Six levels of increasing physical detail:

    mean    – calorimetric ΔT, global h at mean T. U fixed from pump curve Q(I).
    meanH   – same per cooling channel; Q distributed ∝ Sh.
    grad    – U from pressure-drop friction equation, global h at mean T.
    gradH   – same per cooling channel, each sees the common ΔP.
    gradHZ  – per channel + axial power distribution; one mean h per channel.
    gradHZH – same, but one h per axial section (for feelpp boundary conditions).

    The string value matches the ``--cooling`` CLI argument used by feelpp.
    """

    MEAN = "mean"
    MEAN_H = "meanH"
    GRAD = "grad"
    GRAD_H = "gradH"
    GRAD_HZ = "gradHZ"
    GRAD_HZH = "gradHZH"

    @property
    def is_per_channel(self) -> bool:
        """True when calculation is done channel-by-channel (H suffix)."""
        return "H" in self.value

    @property
    def is_axial(self) -> bool:
        """True when axial power distribution is resolved (Z suffix)."""
        return "Z" in self.value

    @property
    def is_mean(self) -> bool:
        """True when U comes from the pump-curve flow rate (non-iterative)."""
        return self.value.startswith("mean")

    @property
    def has_per_section_h(self) -> bool:
        """True when a heat coefficient is required per axial section."""
        return self == CoolingLevel.GRAD_HZH


@dataclass
class ChannelGeometry:
    """
    Geometric parameters for a cooling channel
    
    Attributes:
        hydraulic_diameter: Hydraulic diameter Dh = 4A/P [m]
        cross_section: Cross-sectional area [m²]
        length: Channel length [m]
        name: Channel identifier
    """
    hydraulic_diameter: float  # m
    cross_section: float  # m²
    length: float  # m
    name: str = ""
    
    def __post_init__(self):
        """Validate geometry parameters"""
        if self.hydraulic_diameter <= 0:
            raise ValueError(f"Hydraulic diameter must be positive, got {self.hydraulic_diameter}")
        if self.cross_section <= 0:
            raise ValueError(f"Cross section must be positive, got {self.cross_section}")
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")


@dataclass
class AxialDiscretization:
    """
    Axial discretization for gradHZ cooling mode
    
    Used when power distribution varies along the channel length.
    
    Attributes:
        z_positions: Axial positions [m] (n+1 values for n sections)
        power_distribution: Power per section [W] (n values)
    """
    z_positions: List[float]  # m
    power_distribution: List[float]  # W
    
    def __post_init__(self):
        """Validate discretization"""
        if len(self.z_positions) != len(self.power_distribution) + 1:
            raise ValueError(
                f"z_positions length ({len(self.z_positions)}) must be "
                f"power_distribution length + 1 ({len(self.power_distribution) + 1})"
            )
        
        # Check monotonic increase
        for i in range(len(self.z_positions) - 1):
            if self.z_positions[i] >= self.z_positions[i+1]:
                raise ValueError("z_positions must be strictly increasing")
    
    @property
    def n_sections(self) -> int:
        """Number of axial sections"""
        return len(self.power_distribution)


@dataclass
class ChannelInput:
    """
    Input parameters for thermal-hydraulic calculation of a single channel
    
    Attributes:
        geometry: Channel geometric parameters
        power: Total power dissipated in channel [W]
        temp_inlet: Inlet water temperature [K]
        axial_discretization: Optional axial power distribution
        temp_outlet_guess: Initial guess for outlet temperature [K]
        heat_coeff_guess: Initial guess for heat coefficient [W/m²/K]
        velocity_guess: Initial guess for velocity [m/s]
    """
    geometry: ChannelGeometry
    power: float  # W
    temp_inlet: float  # K
    axial_discretization: Optional[AxialDiscretization] = None
    
    # Initial guesses for faster convergence
    temp_outlet_guess: Optional[float] = None
    heat_coeff_guess: Optional[float] = None
    velocity_guess: Optional[float] = None
    
    def __post_init__(self):
        """Validate input parameters"""
        if self.power < 0:
            raise ValueError(f"Power cannot be negative, got {self.power}")
        if self.temp_inlet <= 0:
            raise ValueError(f"Temperature must be positive (in K), got {self.temp_inlet}")


@dataclass
class ChannelOutput:
    """
    Results from thermal-hydraulic calculation for a single channel
    
    Attributes:
        velocity: Flow velocity [m/s]
        flow_rate: Volumetric flow rate [m³/s]
        friction_factor: Darcy friction factor [dimensionless]
        temp_inlet: Inlet temperature [K]
        temp_outlet: Outlet temperature [K]
        temp_rise: Temperature rise (outlet - inlet) [K]
        temp_mean: Mean temperature [K]
        heat_coeff: Heat transfer coefficient [W/m²/K]
        heat_coeff_distribution: Axial h distribution for gradHZ [W/m²/K]
density_outlet: Water density at outlet [kg/m³]
        specific_heat_outlet: Water specific heat at outlet [kJ/kg/K]
        converged: Whether iteration converged
        iterations: Number of iterations performed
    """
    velocity: float  # m/s
    flow_rate: float  # m³/s
    friction_factor: float  # dimensionless
    
    temp_inlet: float  # K
    temp_outlet: float  # K
    temp_rise: float  # K
    temp_mean: float  # K
    
    heat_coeff: float  # W/m²/K
    # gradHZH only: one h per axial section (n values). None for all other levels.
    heat_coeff_distribution: Optional[List[float]] = None  # W/m²/K per section

    # gradHZ / gradHZH: n per-section temperature rises [dTw_0, …, dTw_{n-1}].
    # feelpp reconstructs local Tw at section k as: T_in + sum(dTw_0..dTw_{k-1}).
    # None for non-axial levels (mean, meanH, grad, gradH).
    temp_rise_distribution: Optional[List[float]] = None  # K per section

    density_outlet: float = 0.0  # kg/m³
    specific_heat_outlet: float = 0.0  # kJ/kg/K

    converged: bool = True
    iterations: int = 0
