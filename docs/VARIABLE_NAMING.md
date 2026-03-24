# Variable Naming Convention for Secondary Cooling Loop

## Overview

This document describes the recommended naming convention for secondary cooling loop variables in the `python_magnetcooling` package.

## Background

High-field magnets use a **two-loop cooling system**:
1. **Primary (closed) cooling loop**: Directly cools the magnet
2. **Secondary (open) cooling loop**: Cools the water in the primary loop via heat exchanger

## Naming Convention

### New Naming (Recommended for CSV files)

For new CSV data files and documentation, use these self-documenting English names:

| Variable | Description | Unit | Old Name |
|----------|-------------|------|----------|
| `flow_secondary` | Secondary cooling loop flow rate | m³/h | `debitbrut` |
| `temp_secondary_in` | Secondary loop inlet temperature | °C | `teb` |
| `temp_secondary_out` | Secondary loop outlet temperature | °C | `tsb` |

**Rationale:**
- **Clear and self-documenting**: No ambiguity about what the variable represents
- **Avoids confusion with time**: `temp_` prefix instead of `t` avoids confusion with time variables
- **Indicates direction**: `_in` and `_out` suffixes clearly show flow direction
- **Language independence**: English terms are more accessible than French abbreviations

### Legacy Naming (For Compatibility)

The following names are maintained in existing code for backward compatibility with historical data:

| Old Name | Origin | Still Used In |
|----------|--------|---------------|
| `debitbrut` | French: "débit brut" (gross flow rate) | API method names, legacy CSV files |
| `teb` | French abbreviation | Legacy data files, some internal code |
| `tsb` | French abbreviation | Legacy data files, some internal code |

## Usage Guidelines

### For New Users

When creating new CSV data files, use the recommended naming:

```csv
time,Pmagnet,flow_secondary,temp_secondary_in,temp_secondary_out
0.0,0.0,100.0,15.0,18.0
1.0,5.0,200.0,15.0,20.0
...
```

### For Code Examples

In documentation and examples, reference both naming conventions:

```python
# Load data with new column names
df = pd.read_csv("data.csv")
power = df["Pmagnet"].values
flow = df["flow_secondary"].values  # Secondary cooling loop flow

# The WaterFlow method keeps the original name for compatibility
flow_rate = waterflow.debitbrut(power)  # Returns secondary flow in m³/h
```

### For API Methods

The `debitbrut()` method name is maintained for backward compatibility:

```python
# Method name uses legacy term
flow_rate = waterflow.debitbrut(power_mw)

# But documentation clarifies it's secondary flow
>>> help(waterflow.debitbrut)
Compute secondary cooling loop flow rate as function of power using hysteresis model.
```

## Migration Path

### Updating Existing Code

When working with legacy data files:

1. **Option 1**: Keep original column names in DataFrame, add comments:
   ```python
   # teb = secondary inlet temp, tsb = secondary outlet temp
   df = pd.read_csv("legacy_data.csv")
   temp_in = df["teb"]
   temp_out = df["tsb"]
   flow = df["debitbrut"]
   ```

2. **Option 2**: Rename columns after loading:
   ```python
   df = pd.read_csv("legacy_data.csv")
   df = df.rename(columns={
       "debitbrut": "flow_secondary",
       "teb": "temp_secondary_in",
       "tsb": "temp_secondary_out"
   })
   ```

### Creating New Data Files

Use the new naming convention directly:

```python
# When exporting new data
df_export = pd.DataFrame({
    "time": times,
    "Pmagnet": power,
    "flow_secondary": flow_rates,
    "temp_secondary_in": temp_in,
    "temp_secondary_out": temp_out
})
df_export.to_csv("new_experiment.csv", index=False)
```

## Summary

| Context | Use | Example |
|---------|-----|---------|
| **New CSV files** | `flow_secondary`, `temp_secondary_in`, `temp_secondary_out` | Recommended |
| **Legacy CSV files** | `debitbrut`, `teb`, `tsb` | Supported |
| **API method names** | `debitbrut()` | For compatibility |
| **Documentation** | Explain both, prefer new naming | This doc |
| **Code comments** | Clarify meaning | `# flow_secondary` |

## References

- Heat exchanger module: [heatexchanger_primary.py](../python_magnetcooling/heatexchanger_primary.py)
- Hysteresis documentation: [debitbrut_hysteresis.md](debitbrut_hysteresis.md)
- Example usage: [waterflow_debitbrut_example.py](../examples/waterflow_debitbrut_example.py)
