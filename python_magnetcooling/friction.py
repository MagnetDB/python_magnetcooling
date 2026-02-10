"""
Friction factor models for internal flow.

Implements various friction factor correlations:
- Constant: Fixed friction factor
- Blasius: Smooth pipe turbulent flow
- Filonenko: Improved smooth pipe correlation
- Colebrook-White: Universal correlation with roughness
- Swamee-Jain: Explicit approximation of Colebrook
"""

from abc import ABC, abstractmethod
from typing import Dict, Type
from math import exp, log, log10, sqrt
from .exceptions import FrictionError


class FrictionModel(ABC):
    """Base class for friction factor models"""
    
    def __init__(self, roughness: float = 0.012e-3):
        """
        Initialize friction model
        
        Args:
            roughness: Surface roughness [m] (default: 0.012 mm for drawn copper)
        """
        self.roughness = roughness
    
    @abstractmethod
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """
        Compute friction factor
        
        Args:
            reynolds: Reynolds number
            hydraulic_diameter: Hydraulic diameter [m]
            friction_guess: Initial guess for iterative methods
            
        Returns:
            Friction factor [dimensionless]
        """
        pass


class ConstantFriction(FrictionModel):
    """Constant friction factor (no dependence on Re)"""
    
    def __init__(self, value: float = 0.055, **kwargs):
        """
        Initialize with constant value
        
        Args:
            value: Friction factor value (default: 0.055)
        """
        super().__init__(**kwargs)
        self.value = value
    
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """Return constant friction factor"""
        return self.value


class BlasiusFriction(FrictionModel):
    """
    Blasius correlation for smooth pipes
    
    f = 0.316 / Re^0.25
    Valid for: Re < 100,000
    """
    
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """Compute friction factor using Blasius"""
        
        if reynolds < 2300:
            # Laminar flow: f = 64/Re
            return 64.0 / reynolds
        
        return 0.316 / exp(log(reynolds) * 0.25)


class FilonenkoFriction(FrictionModel):
    """
    Filonenko correlation for smooth pipes
    
    f = 1 / (1.82·log₁₀(Re) - 1.64)²
    Valid for: 10⁴ < Re < 10⁶
    """
    
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """Compute friction factor using Filonenko"""
        
        if reynolds < 2300:
            return 64.0 / reynolds
        
        return 1.0 / (1.82 * log10(reynolds) - 1.64) ** 2


class ColebrookFriction(FrictionModel):
    """
    Colebrook-White equation (implicit)
    
    1/√f = -2·log₁₀(ε/(3.7·D) + 2.51/(Re·√f))
    
    Solved iteratively. Universal correlation valid for all regimes.
    """
    
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """Compute friction factor using Colebrook (iterative)"""
        
        if reynolds < 2300:
            return 64.0 / reynolds
        
        # Relative roughness
        rel_roughness = self.roughness / hydraulic_diameter
        
        # Iterative solution
        f = friction_guess
        max_iter = 20
        tolerance = 1e-6
        
        for i in range(max_iter):
            f_new = 1.0 / (
                -2.0 * log10(rel_roughness / 3.7 + 2.51 / (reynolds * sqrt(f)))
            ) ** 2
            
            error = abs(f_new - f) / f
            f = f_new
            
            if error < tolerance:
                return f
        
        raise FrictionError(
            f"Colebrook iteration did not converge after {max_iter} iterations"
        )


class SwameeFriction(FrictionModel):
    """
    Swamee-Jain equation (explicit approximation of Colebrook)
    
    f = 1.325 / [ln(ε/(3.7·D) + 5.74/Re^0.9)]²
    
    Accurate to within 1% of Colebrook. Non-iterative.
    """
    
    def compute(
        self,
        reynolds: float,
        hydraulic_diameter: float,
        friction_guess: float = 0.055
    ) -> float:
        """Compute friction factor using Swamee-Jain"""
        
        if reynolds < 2300:
            return 64.0 / reynolds
        
        # Relative roughness
        rel_roughness = self.roughness / hydraulic_diameter
        
        from math import log as ln
        
        f = 1.325 / (
            ln(rel_roughness / 3.7 + 5.74 / exp(log(reynolds) * 0.9))
        ) ** 2
        
        return f


# Registry of available friction models
_FRICTION_MODELS: Dict[str, Type[FrictionModel]] = {
    "Constant": ConstantFriction,
    "Blasius": BlasiusFriction,
    "Filonenko": FilonenkoFriction,
    "Colebrook": ColebrookFriction,
    "Swamee": SwameeFriction,
}


def get_friction_model(
    name: str,
    roughness: float = 0.012e-3,
    constant_value: float = 0.055
) -> FrictionModel:
    """
    Get friction model by name
    
    Args:
        name: Model name (Constant, Blasius, Filonenko, Colebrook, Swamee)
        roughness: Surface roughness [m]
        constant_value: Value for Constant model
        
    Returns:
        Friction model instance
        
    Raises:
        FrictionError: If model name is unknown
    """
    if name not in _FRICTION_MODELS:
        raise FrictionError(
            f"Unknown friction model '{name}'. Available: {list(_FRICTION_MODELS.keys())}"
        )
    
    if name == "Constant":
        return _FRICTION_MODELS[name](value=constant_value)
    else:
        return _FRICTION_MODELS[name](roughness=roughness)


def available_friction_models() -> list[str]:
    """Get list of available friction models"""
    return list(_FRICTION_MODELS.keys())
