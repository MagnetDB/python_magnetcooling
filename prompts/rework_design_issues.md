# Rework `python_magnetcooling` — fix duplicate dataclasses and wire OOP correlations/friction into the solver

The `python_magnetcooling` package has two structural issues that need to be resolved.

---

## Issue 1 — Duplicate dataclasses (bug)

`ChannelGeometry`, `AxialDiscretization`, `ChannelInput`, and `ChannelOutput` are defined twice:
- in `python_magnetcooling/channel.py` (the public API, exported via `__init__.py`)
- in `python_magnetcooling/thermohydraulics.py` (private copies used by the solver)

`__init__.py` lines 54–59 export the `channel.py` versions. But `thermohydraulics.py` never imports from `channel.py` — it uses its own definitions. This means that objects created by an external caller and objects handled internally by the solver are instances of different classes, breaking `isinstance` checks and type annotations.

Additional differences:
- `channel.py`'s `AxialDiscretization` enforces monotonic `z_positions` (lines 58–60) and exposes a `n_sections` property (lines 62–65).
- `channel.py`'s `ChannelInput` validates `power >= 0` and `temp_inlet > 0` in `__post_init__`. Neither guard exists in the `thermohydraulics.py` copies.

**Fix:** Remove the four dataclass definitions from `thermohydraulics.py` and replace them with:

```python
from .channel import ChannelGeometry, AxialDiscretization, ChannelInput, ChannelOutput
```

---

## Issue 2 — OOP `correlations.py` / `friction.py` are dead code (architectural debt)

`correlations.py` provides a `HeatCorrelation` ABC with concrete implementations:
- `MontgomeryCorrelation`
- `DittusBoelterCorrelation`
- `ColburnCorrelation`
- `SilverbergCorrelation`
- `get_correlation(name, fuzzy_factor)` factory

`friction.py` provides a `FrictionModel` ABC with concrete implementations:
- `ConstantFriction`
- `BlasiusFriction`
- `FilonenkoFriction`
- `ColebrookFriction`
- `SwameeFriction`
- `get_friction_model(name, roughness)` factory

Both are exported from `__init__.py` (lines 67–68) as part of the public API.

However, `thermohydraulics.py` line 13 imports only from `cooling.py`:

```python
from .cooling import steam, Uw, getDT, getHeatCoeff, getTout
```

The solver calls `getHeatCoeff()` and `Uw()` from `cooling.py`, which internally dispatch through plain function dicts. The OOP classes are never called — they are isolated dead code paths.

**Fix:** Refactor `_compute_channel_uniform()` and `_compute_channel_axial()` in `thermohydraulics.py` to instantiate correlations and friction models via `get_correlation()` and `get_friction_model()` at the start of `compute()`, then call `.compute()` on them inside the iteration loop. The `getHeatCoeff` and `Uw` imports from `cooling.py` can then be removed. After this, `cooling.py` can be reduced to just `steam`, `getDT`, and `getTout` (the functions that have no OOP equivalent yet), or eliminated entirely if those are also migrated to `WaterProperties`.

---

## Scope and constraints

- Do not change the public API (`__init__.py` `__all__` list must remain the same).
- All existing tests under `tests/` must continue to pass after each change.
- Treat the two issues as independent commits: fix Issue 1 first (small, safe), then Issue 2.
- The `feelpp.py` module may import from `cooling.py` directly — check before removing anything from it.
