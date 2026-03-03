# Friction Factor Models

## Overview

The `python_magnetcooling.friction` module provides various friction factor correlations for internal flow calculations.

## Available Models

### 1. Constant Friction
Fixed friction factor, independent of Reynolds number.

```python
from python_magnetcooling.friction import get_friction_model

model = get_friction_model('Constant', constant_value=0.055)
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

### 2. Blasius Correlation
For smooth pipes in turbulent flow: `f = 0.316 / Re^0.25`

Valid range: Re < 100,000

```python
model = get_friction_model('Blasius')
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

### 3. Filonenko Correlation
Improved correlation for smooth pipes: `f = 1 / (1.82·log₁₀(Re) - 1.64)²`

Valid range: 10⁴ < Re < 10⁶

```python
model = get_friction_model('Filonenko')
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

### 4. Karman-Nikuradse Correlation (NEW)
More accurate than Blasius for high Reynolds numbers.

Implicit equation: `1/√f = 1.93·log₁₀(Re·√f) - 0.537`

Valid range: Re > 10⁴

```python
model = get_friction_model('Karman')
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

**Features:**
- Iterative solution with automatic convergence
- More accurate than Blasius at high Reynolds numbers
- Smooth pipe correlation

### 5. Colebrook-White Equation
Universal correlation with surface roughness (implicit).

```python
model = get_friction_model('Colebrook', roughness=0.012e-3)
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

### 6. Swamee-Jain Equation
Explicit approximation of Colebrook (non-iterative).

```python
model = get_friction_model('Swamee', roughness=0.012e-3)
f = model.compute(reynolds=50000, hydraulic_diameter=0.008)
```

## Usage Examples

### Basic Usage

```python
from python_magnetcooling.friction import get_friction_model

# Operating conditions
reynolds = 50000
hydraulic_diameter = 0.008  # 8 mm

# Compare different models
models = ['Blasius', 'Filonenko', 'Karman', 'Colebrook']

for model_name in models:
    model = get_friction_model(model_name)
    f = model.compute(reynolds, hydraulic_diameter)
    print(f"{model_name:12s}: f = {f:.6f}")
```

### With Surface Roughness

```python
from python_magnetcooling.friction import get_friction_model

# Copper tube with typical roughness
roughness = 0.012e-3  # 0.012 mm (default for drawn copper)

# Compare smooth vs rough pipe models
smooth = get_friction_model('Karman')
rough = get_friction_model('Colebrook', roughness=roughness)

reynolds = 50000
dh = 0.008

f_smooth = smooth.compute(reynolds, dh)
f_rough = rough.compute(reynolds, dh)

print(f"Smooth pipe (Karman):     f = {f_smooth:.6f}")
print(f"Rough pipe (Colebrook):   f = {f_rough:.6f}")
print(f"Difference: {(f_rough/f_smooth - 1)*100:.2f}%")
```

### Integration with Thermal-Hydraulic Solver

```python
from python_magnetcooling import ThermalHydraulicInput, ThermalHydraulicCalculator
from python_magnetcooling.friction import get_friction_model

# Use Karman model in thermal-hydraulic calculation
inputs = ThermalHydraulicInput(
    channels=[...],
    pressure_inlet=15.0,
    pressure_drop=5.0,
    friction_model="Karman",  # Specify friction model by name
    heat_correlation="Montgomery"
)

calculator = ThermalHydraulicCalculator()
result = calculator.compute(inputs)
```

## Choosing the Right Model

| Model | Use Case | Advantages | Limitations |
|-------|----------|------------|-------------|
| **Constant** | Quick estimates, validation | Simple, fast | Not physically accurate |
| **Blasius** | Smooth pipes, moderate Re | Simple explicit formula | Only valid Re < 100,000 |
| **Filonenko** | Smooth pipes, higher Re | Better than Blasius | Valid 10⁴ < Re < 10⁶ |
| **Karman** | Smooth pipes, high accuracy | More accurate than Blasius | Requires iteration |
| **Colebrook** | Rough pipes, universal | Most accurate, all regimes | Requires iteration |
| **Swamee** | Rough pipes, non-iterative | Fast, explicit | Slight approximation |

## Validation and Accuracy

### Karman vs Blasius Comparison

For smooth pipes at Re = 50,000:
- Blasius: f ≈ 0.0211
- Karman: f ≈ 0.0196
- Difference: ~7%

The Karman correlation is generally more accurate for high Reynolds numbers.

## References

1. Blasius, H. (1913). "Das Aehnlichkeitsgesetz bei Reibungsvorgängen in Flüssigkeiten"
2. Filonenko, G. K. (1954). "Hydraulic resistance in pipes"
3. Karman, T. von (1930). "Mechanische Ähnlichkeit und Turbulenz"
4. Colebrook, C. F. (1939). "Turbulent flow in pipes"

## See Also

- [Correlations Module](correlations.md) - Heat transfer correlations (including Gnielinski)
- [Water Properties](water_properties.md) - IAPWS-IF97 implementation
- [Thermal-Hydraulics](thermohydraulics.md) - Main solver documentation
