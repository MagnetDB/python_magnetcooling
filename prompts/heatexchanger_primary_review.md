# Code Review: heatexchanger_primary.py

**Date:** February 9, 2026  
**File:** `python_magnetcooling/heatexchanger_primary.py`  
**Lines of Code:** 1,140

---

## Executive Summary

The module provides heat exchanger calculations for primary cooling loops with good documentation and type hints. However, it has several critical issues including missing imports, type inconsistencies, and extensive code duplication that should be addressed.

---

## ✅ Strengths

1. **Good Documentation**: Most functions have clear docstrings with parameter descriptions
2. **Type Hints**: Many functions use type annotations for better code clarity
3. **Custom Exception Handling**: Dedicated exceptions for validation errors (`InvalidNTUError`, `InvalidHeatTransferError`, `InvalidTemperatureError`)
4. **Logical Organization**: Well-separated calculation and plotting functions
5. **Property Wrapper Functions**: Backward-compatible wrappers for water properties

---

## 🔴 Critical Issues

### 1. Missing Import (Line 25)
```python
import ht
```
**Problem:** Module `ht` could not be resolved  
**Impact:** Code will not run  
**Fix:** Install `ht` package or add to `pyproject.toml` dependencies

```bash
pip install ht
```

### 2. Type Inconsistency - String "None" Instead of Optional (Multiple locations)

**Affected Functions:**
- `calculate_heat_profiles()` (line 132)
- `plot_heat_balance()` (line 204)
- `display_Q()` (line 295)
- `calculate_temperature_profiles()` (line 366)
- `display_T()` (line 493)
- `main()` (line 964)

**Current (Problematic):**
```python
def calculate_heat_profiles(
    mrun,
    debit_alim: float,
    ohtc: float | str,  # ❌ Using string "None"
) -> pd.DataFrame:
    ...
    if ohtc != "None":  # ❌ String comparison
```

**Should Be:**
```python
from typing import Optional

def calculate_heat_profiles(
    mrun,
    debit_alim: float,
    ohtc: Optional[float],  # ✅ Proper type
) -> pd.DataFrame:
    ...
    if ohtc is not None:  # ✅ Proper None check
```

**Impact:** Type safety issues, confusing API, potential bugs  
**Priority:** High

### 3. Hardcoded Magic Numbers

**Line 726:**
```python
A = 1063.4  # m^2
```

**Lines 812-814:**
```python
Nc = int((553 - 1) / 2.0)  # (Number of plates -1)/2
Ac = 3.0e-3 * 1.174  # Plate spacing * Plate width [m²]
de = 2 * 3.0e-3  # 2*Plate spacing [m]
```

**Recommended:**
```python
# Module-level constants at the top
# Heat exchanger geometry constants
HEAT_EXCHANGER_AREA = 1063.4  # m²
NUMBER_OF_PLATES = 553
PLATE_SPACING = 3.0e-3  # m
PLATE_WIDTH = 1.174  # m
PLATE_CHANNEL_AREA = PLATE_SPACING * PLATE_WIDTH  # m²
HYDRAULIC_DIAMETER = 2 * PLATE_SPACING  # m
NUMBER_OF_CHANNELS = int((NUMBER_OF_PLATES - 1) / 2.0)
```

**Impact:** Maintainability, testability  
**Priority:** High

---

## ⚠️ Major Issues

### 4. Inconsistent Naming Convention (Line 121)
```python
mixingTemp = mixing_temp
```

**Problem:** Maintains both snake_case and camelCase names  
**Fix:** Remove backward compatibility alias or clearly mark as deprecated  
**Priority:** Medium

### 5. Extensive Code Duplication

**Display Functions:** Lines 336-356 vs 533-545 contain identical plotting code

**Example (duplicated in multiple places):**
```python
# In display_Q()
fig, ax = plt.subplots()
df.plot(x="t", y="Qhot", ax=ax)
df.plot(x="t", y="Pmagnet", ax=ax)
df.plot(x="t", y="Ptot", ax=ax)
plt.ylabel(r"Q[MW]")
plt.xlabel(r"t [s]")
plt.grid(True)
# ... repeated later for saving

# In display_T() - similar pattern repeated
```

**Recommendation:** Create single plotting function that handles both display and save:
```python
def _create_and_save_plot(df, x, y_cols, ylabel, xlabel, title, 
                          save_path=None, show=False):
    fig, ax = plt.subplots()
    for y in y_cols:
        df.plot(x=x, y=y, ax=ax)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True)
    
    if show:
        plt.show()
    elif save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
```

**Impact:** Maintainability, bug risk  
**Lines Affected:** ~150 lines could be reduced  
**Priority:** High

### 6. Overly Long Function - main() (Lines 949-1137)

**Current Length:** ~190 lines  
**Issues:**
- Multiple responsibilities
- Hard to test
- Poor readability

**Recommendation:** Break into smaller functions:
```python
def parse_arguments(command_line=None):
    """Parse command line arguments."""
    ...

def load_and_preprocess_data(args, mrun):
    """Load data and apply filtering/smoothing."""
    ...

def calculate_all_fields(df, args):
    """Calculate temperature and heat transfer fields."""
    ...

def generate_plots(df, mrun, args):
    """Generate and save all plots."""
    ...

def main(command_line=None):
    """Main entry point."""
    args = parse_arguments(command_line)
    mrun = load_and_preprocess_data(args)
    df = calculate_all_fields(mrun, args)
    generate_plots(df, mrun, args)
```

**Priority:** Medium

### 7. Dead Code (Lines 604-614)
```python
# def heatBalance(Tin, Pin, Debit, Power, debug=False):
#    """
#    Computes Tout from heatBalance
#    ...
```

**Fix:** Remove commented code (use version control history if needed)  
**Priority:** Low

### 8. Missing Type Hints

**Functions without return types:**
- `estimate_temperature_elevation()` (line 569) - missing `-> float`
- `main()` (line 949) - missing parameter type for `command_line`

**Recommendation:**
```python
def estimate_temperature_elevation(
    power: float, 
    flow_rate: float, 
    inlet_temp: float, 
    outlet_pressure: float, 
    inlet_pressure: float, 
    iterations: int = 10
) -> float:  # ✅ Add return type
    ...

def main(command_line: Optional[list[str]] = None) -> None:  # ✅ Add types
    ...
```

**Priority:** Medium

### 9. Unsafe Exception Handling (Lines 16-18)
```python
try:
    matplotlib.rcParams["text.usetex"] = True
except Exception:  # ❌ Too broad
    pass
```

**Better:**
```python
try:
    matplotlib.rcParams["text.usetex"] = True
except (KeyError, RuntimeError) as e:  # ✅ Specific exceptions
    # LaTeX not available, use default rendering
    pass
```

**Priority:** Low

---

## 💡 Recommendations

### 10. Complex Lambda Functions (Lines 138-144, and many others)

**Current:**
```python
df["Qhot"] = df.apply(
    lambda row: ((row.Flow) * 1.0e-3 + 0 / 3600.0)
    * (
        get_rho(row.BP, row.Tout) * get_cp(row.BP, row.Tout) * (row.Tout)
        - get_rho(row.HP, row.Tin) * get_cp(row.HP, row.Tin) * row.Tin
    )
    / 1.0e6,
    axis=1,
)
```

**Better:**
```python
def calculate_qhot(row):
    """Calculate heat from flow and temperature change."""
    flow = row.Flow * 1.0e-3
    rho_cp_out = get_rho(row.BP, row.Tout) * get_cp(row.BP, row.Tout)
    rho_cp_in = get_rho(row.HP, row.Tin) * get_cp(row.HP, row.Tin)
    return flow * (rho_cp_out * row.Tout - rho_cp_in * row.Tin) / 1.0e6

df["Qhot"] = df.apply(calculate_qhot, axis=1)
```

**Benefits:**
- Easier to test
- Better readability
- Can add proper type hints

**Priority:** Medium

### 11. Add Input Validation

Functions like `heatexchange()` should validate inputs:

```python
def heatexchange(
    h: float,
    Tci: float,
    Thi: float,
    Debitc: float,
    Debith: float,
    Pci: float,
    Phi: float,
    debug: bool = False,
) -> Tuple[float, float, float]:
    """..."""
    # Add validation
    if h <= 0:
        raise ValueError(f"Heat transfer coefficient must be positive, got {h}")
    if Debitc <= 0 or Debith <= 0:
        raise ValueError("Flow rates must be positive")
    if Pci <= 0 or Phi <= 0:
        raise ValueError("Pressures must be positive")
    if Tci < -273.15 or Thi < -273.15:
        raise ValueError("Temperatures below absolute zero")
    
    # ... rest of function
```

**Priority:** Medium

### 12. Improve Variable Names

**Current → Suggested:**
- `h` → `heat_transfer_coeff`
- `A` → `heat_exchanger_area`
- `de` → `hydraulic_diameter`
- `Tci` → `cold_inlet_temp`
- `Thi` → `hot_inlet_temp`
- `Debitc` → `cold_flow_rate`
- `Debith` → `hot_flow_rate`

**Priority:** Low (breaking change)

### 13. Inconsistent Unit Comments

Some functions have good unit documentation:
```python
def get_rho(pressure: float, temperature: float) -> float:
    """Get water density (kg/m³). Args: pressure (bar), temperature (°C)."""
```

But `heatexchange()` only has comments in docstring. Consider adding to signature:
```python
def heatexchange(
    h: float,  # W/m²/K
    Tci: float,  # °C
    Thi: float,  # °C
    Debitc: float,  # m³/h
    Debith: float,  # l/s
    Pci: float,  # bar
    Phi: float,  # bar
    debug: bool = False,
) -> Tuple[float, float, float]:  # (Tco °C, Tho °C, Q W)
```

**Priority:** Low

---

## 📊 Code Metrics

- **Total Lines:** 1,140
- **Functions:** 24
- **Average Function Length:** ~47 lines (main() skews this heavily)
- **Estimated Code Duplication:** ~15-20%
- **Potential Line Reduction:** ~150-200 lines (through refactoring)

---

## 🎯 Priority Action Items

### Immediate (Critical)
1. ✅ **Install `ht` package** - Code won't run without it
2. ✅ **Replace string "None" with Optional[float]** - Type safety issue

### Short Term (High Priority)
3. ✅ **Extract magic numbers to constants** - Maintainability
4. ✅ **Remove code duplication in display functions** - DRY principle
5. ✅ **Remove dead code** - Code cleanliness

### Medium Term
6. ✅ **Break down main() function** - Testability
7. ✅ **Extract complex lambdas to named functions** - Readability
8. ✅ **Add missing type hints** - Type safety
9. ✅ **Add input validation** - Robustness

### Long Term (Nice to Have)
10. ✅ **Improve variable names** - Requires API change
11. ✅ **Enhance exception handling** - Better error messages
12. ✅ **Consistent unit documentation** - Better UX

---

## 📝 Testing Recommendations

Currently no tests visible. Recommend adding:

1. **Unit tests** for core calculations:
   - `calculate_heat_capacity_and_density()`
   - `calculate_mass_flow_rates()`
   - `mixing_temp()`
   - `estimate_temperature_elevation()`

2. **Integration tests** for:
   - `heatexchange()` - main heat exchanger model
   - Complete workflow through `calculate_heat_profiles()`

3. **Validation tests**:
   - Test that custom exceptions are raised appropriately
   - Test boundary conditions (zero flow, extreme temps)

4. **Mock external dependencies** (`ht` package, `pandas`, `matplotlib`)

---

## 🔧 Suggested Dependencies Check

Verify these are in `pyproject.toml`:
- `ht` - Heat transfer library
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `python_magnetrun` - Custom dependency
- Water properties library (IAPWS?)

---

## Summary

The code is functional and well-documented but would benefit significantly from:
1. Addressing the critical import issue
2. Improving type safety
3. Reducing code duplication
4. Breaking down complex functions
5. Adding comprehensive tests

**Overall Grade: B-**  
*With recommended changes: A-*
