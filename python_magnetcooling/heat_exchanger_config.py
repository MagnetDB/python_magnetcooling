"""Heat exchanger configuration and geometry specifications."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class HeatExchangerConfig:
    """Configuration and geometry for plate heat exchanger.

    Attributes:
        area: Total heat transfer area (m²)
        num_plates: Total number of plates
        plate_spacing: Gap between plates (m)
        plate_width: Width of each plate (m)
        correlation_params: Heat transfer correlation parameters [a, b, c] for Nu = a * Re^b * Pr^c
        use_nominal_params: If True, use nominal correlation parameters instead of fitted
    """

    # Heat exchanger geometry from datasheet
    area: float = 1063.4  # m²
    num_plates: int = 553
    plate_spacing: float = 3.0e-3  # m (3 mm)
    plate_width: float = 1.174  # m

    # Heat transfer correlation parameters
    # Default: from student parameter fits
    correlation_params: List[float] = field(default_factory=lambda: [0.1249, 0.65453, 0.40152])

    # Alternative parameter sets
    use_nominal_params: bool = False

    def __post_init__(self):
        """Set correlation parameters based on configuration."""
        if self.use_nominal_params:
            # Nominal values from heat exchanger specifications
            self.correlation_params = [0.207979, 0.640259, 0.397994]

    @property
    def num_channels(self) -> int:
        """Number of flow channels: (num_plates - 1) / 2."""
        return int((self.num_plates - 1) / 2.0)

    @property
    def channel_area(self) -> float:
        """Cross-sectional area of one channel (m²)."""
        return self.plate_spacing * self.plate_width

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter: 2 * plate_spacing (m)."""
        return 2.0 * self.plate_spacing

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"HeatExchangerConfig(\n"
            f"  area={self.area} m²,\n"
            f"  num_plates={self.num_plates},\n"
            f"  num_channels={self.num_channels},\n"
            f"  plate_spacing={self.plate_spacing*1000:.1f} mm,\n"
            f"  plate_width={self.plate_width} m,\n"
            f"  channel_area={self.channel_area*1e6:.2f} cm²,\n"
            f"  hydraulic_diameter={self.hydraulic_diameter*1000:.1f} mm,\n"
            f"  correlation_params={self.correlation_params}\n"
            f")"
        )


# Default configuration instance
DEFAULT_HX_CONFIG = HeatExchangerConfig()


# Alternative configurations
NOMINAL_HX_CONFIG = HeatExchangerConfig(use_nominal_params=True)
