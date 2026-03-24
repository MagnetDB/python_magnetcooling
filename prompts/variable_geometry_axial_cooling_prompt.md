# Implementation: Variable-Geometry Axial Cooling (`gradHZHvar`)

## Context

The package already implements six cooling levels up to `gradHZH` (per-channel,
axial power distribution, per-section heat coefficient).  This prompt adds the
next level: **`gradHZHvar`** — all of the above, plus hydraulic diameter `Dh`
and hydraulic cross-section `Sh` that vary **per axial section**.

### Why variable geometry matters

Magnet coils can have grooves, ribs, or varying slot widths along their length.
The existing axial solver (`_compute_channel_axial`) assumes a single `Dh` and
`Sh` for the whole channel.  With variable geometry:

- **`Dh_k`** changes the heat coefficient and friction factor at each section.
- **`Sh_k`** changes the local flow velocity through mass conservation:
  `U_k = ṁ / (ρ_k · Sh_k)`, where `ṁ` is the (constant) mass flow rate.
- The **pressure drop** must be accumulated section-by-section:
  `ΔP = Σ_k  cf_k · (ρ_k / 2) · U_k² · (L_k / Dh_k)`

### Tw output convention (already established)

For all axial levels (including `gradHZHvar`) the Tw output is **`dTw` per
section per channel** — i.e. `ChannelOutput.temp_rise_distribution`.  feelpp
reconstructs the local water temperature at section `k` as:

    T_w(k) = T_in + sum(dTw[0], …, dTw[k-1])

`temp_distribution` (n+1 boundary temperatures) is **not** stored in
`ChannelOutput`; it is only a local variable inside the solver.

---

## 1. Changes to `channel.py`

### 1.1 New `CoolingLevel` entry

Add one new enum value after `GRAD_HZH`:

```python
GRAD_HZH_VAR = "gradHZHvar"
```

Update the class docstring to include:

```
gradHZHvar – same as gradHZH but Dh and Sh can vary per axial section.
```

Update `has_per_section_h`:

```python
@property
def has_per_section_h(self) -> bool:
    return self in (CoolingLevel.GRAD_HZH, CoolingLevel.GRAD_HZH_VAR)
```

Add a new property:

```python
@property
def has_variable_geometry(self) -> bool:
    """True when Dh and Sh may vary by axial section."""
    return self == CoolingLevel.GRAD_HZH_VAR
```

The existing `is_per_channel` and `is_axial` properties remain correct because
`"gradHZHvar"` contains `"H"` and `"Z"`.

### 1.2 Extend `AxialDiscretization`

Add two optional fields for per-section geometry (n values, one per section,
matching `power_distribution`):

```python
@dataclass
class AxialDiscretization:
    z_positions: List[float]          # m, n+1 values
    power_distribution: List[float]   # W, n values

    # gradHZHvar only: per-section hydraulic geometry.
    # None for all other cooling levels.
    hydraulic_diameters: Optional[List[float]] = None   # m, n values
    cross_sections: Optional[List[float]] = None        # m², n values

    def __post_init__(self):
        # existing validation (length checks, monotonic z) …

        # New validation when variable geometry is supplied:
        if self.hydraulic_diameters is not None:
            if len(self.hydraulic_diameters) != self.n_sections:
                raise ValueError(
                    "hydraulic_diameters must have one entry per section "
                    f"(expected {self.n_sections}, got {len(self.hydraulic_diameters)})"
                )
        if self.cross_sections is not None:
            if len(self.cross_sections) != self.n_sections:
                raise ValueError(
                    "cross_sections must have one entry per section "
                    f"(expected {self.n_sections}, got {len(self.cross_sections)})"
                )
        if (self.hydraulic_diameters is None) != (self.cross_sections is None):
            raise ValueError(
                "hydraulic_diameters and cross_sections must both be supplied "
                "or both be None"
            )
```

### 1.3 Extend `ChannelOutput`

Add one new optional field for the per-section velocity profile (useful for
feelpp post-processing and debugging):

```python
# gradHZHvar only: per-section mean velocity U_k = ṁ/(ρ_k·Sh_k).  [m/s]
velocity_distribution: Optional[List[float]] = None
```

The existing `velocity` field retains its meaning: the mass-averaged velocity
`ṁ / (ρ_mean · Sh_mean)` where `Sh_mean = mean(Sh_k)`.

Update the `ChannelOutput` docstring accordingly.

---

## 2. Changes to `thermohydraulics.py`

### 2.1 `_validate_inputs`

Extend the axial validation block to check that `gradHZHvar` channels supply
variable geometry in their `AxialDiscretization`:

```python
if inputs.cooling_level.has_variable_geometry:
    for ch in inputs.channels:
        if ch.axial_discretization is None:
            raise ValidationError("gradHZHvar requires axial_discretization.")
        if ch.axial_discretization.hydraulic_diameters is None:
            raise ValidationError(
                "gradHZHvar requires axial_discretization.hydraulic_diameters "
                "and .cross_sections."
            )
```

### 2.2 Routing in `compute()`

In the main dispatch block, route `GRAD_HZH_VAR` to the new solver:

```python
elif cooling_level.is_axial:
    if cooling_level.has_variable_geometry:
        ch_out = self._compute_channel_axial_var(ch_input, inputs)
    else:
        ch_out = self._compute_channel_axial(ch_input, inputs)
```

### 2.3 New method `_compute_channel_axial_var`

This is the core of the implementation.  Structure it analogously to
`_compute_channel_axial` but replace the single `geom.hydraulic_diameter` /
`geom.cross_section` with per-section values.

```python
def _compute_channel_axial_var(
    self,
    channel: ChannelInput,
    inputs: ThermalHydraulicInput,
) -> ChannelOutput:
    """
    Axial-marching solver for ``gradHZHvar``.

    Hydraulic diameter Dh_k and cross-section Sh_k may differ per section.
    Mass flow rate ṁ is conserved; local velocity U_k = ṁ / (ρ_k · Sh_k).
    Pressure drop is the sum of per-section contributions.

    Tw output: temp_rise_distribution = [dTw_0, …, dTw_{n-1}].
    feelpp reconstructs T_w(k) = T_in + sum(dTw[0..k-1]).
    """
    geom  = channel.geometry
    axial = channel.axial_discretization
    n     = axial.n_sections
    Dh_k  = axial.hydraulic_diameters   # List[float], n values
    Sh_k  = axial.cross_sections         # List[float], n values

    # Initial boundary temperatures (n+1 values, local variable only).
    T_z = [channel.temp_inlet] + [channel.temp_inlet + 10.0] * n
    h_z = [80_000.0] * n
    U_z = [5.0] * n      # per-section velocity (output for feelpp)

    # Initial mass flow rate guess using channel-mean geometry.
    Sh_mean = sum(Sh_k) / n
    Dh_mean = sum(Dh_k) / n
    T_mean_init = channel.temp_inlet + 5.0
    state0 = WaterProperties.get_state(T_mean_init, inputs.pressure_inlet)
    m_dot = state0.density * (
        channel.velocity_guess if channel.velocity_guess is not None else 5.0
    ) * Sh_mean

    cf_k = [0.055] * n
    converged = False
    iteration = 0

    z0     = axial.z_positions[0]
    z_span = axial.z_positions[-1] - z0

    for iteration in range(inputs.max_iterations):
        T_z_old = T_z.copy()
        m_dot_old = m_dot

        # ── Temperature marching ──────────────────────────────────────────
        for k in range(n):
            z_frac  = (axial.z_positions[k] - z0) / z_span
            P_local = inputs.pressure_inlet - inputs.pressure_drop * z_frac

            # Local density at current section midpoint temperature.
            T_mean_k = (T_z[k] + T_z[k + 1]) / 2.0
            state_k  = WaterProperties.get_state(T_mean_k, P_local)

            # Local volumetric flow rate and velocity.
            Q_k   = m_dot / state_k.density
            U_z[k] = Q_k / Sh_k[k]

            # Calorimetric temperature rise for this section.
            dT_k    = getDT(Q_k, axial.power_distribution[k], T_mean_k, P_local)
            T_z[k + 1] = T_z[k] + dT_k

        # ── Heat coefficient per section ──────────────────────────────────
        for k in range(n):
            T_mid_k = (T_z[k] + T_z[k + 1]) / 2.0
            z_frac  = (axial.z_positions[k] - z0) / z_span
            P_local = inputs.pressure_inlet - inputs.pressure_drop * z_frac
            h_z[k]  = self._correlation.compute(
                T_mid_k, P_local, U_z[k], Dh_k[k], geom.length
            )

        # ── Update mass flow rate from section-summed pressure drop ───────
        # ΔP_total = Σ_k  cf_k · (ρ_k/2) · U_k² · (L_k / Dh_k)
        # where L_k = z_positions[k+1] - z_positions[k].
        T_mean = (T_z[0] + T_z[-1]) / 2.0
        state_mean = WaterProperties.get_state(T_mean, inputs.pressure_inlet)

        # Solve for new m_dot by Newton step or bisection on the ΔP residual.
        m_dot_new, cf_k = self._solve_mass_flow_axial_var(
            m_dot, T_z, axial, Dh_k, Sh_k, inputs, geom.length
        )

        # Convergence check.
        err_flow = abs(1.0 - m_dot_new / m_dot) if m_dot > 0 else 1.0
        err_temp = max(
            abs(1.0 - T_z[k + 1] / T_z_old[k + 1])
            for k in range(n)
            if T_z_old[k + 1] != 0.0
        )

        if self.verbose:
            print(
                f"  Iter {iteration}: ṁ={m_dot_new:.4f}, T_out={T_z[-1]:.3f}, "
                f"err_flow={err_flow:.2e}, err_temp={err_temp:.2e}"
            )

        m_dot = m_dot_new

        if err_flow < inputs.tolerance_flow and err_temp < inputs.tolerance_temp:
            converged = True
            break

    state_out = WaterProperties.get_state(T_z[-1], inputs.pressure_inlet)

    # Representative velocity (mass-average equivalent at mean cross-section).
    U_mean = m_dot / (state_out.density * Sh_mean)
    # Representative friction factor (length-weighted mean).
    L_k = [axial.z_positions[k + 1] - axial.z_positions[k] for k in range(n)]
    cf_mean = sum(cf_k[k] * L_k[k] for k in range(n)) / sum(L_k)

    # Tw output: dTw per section.  feelpp reconstructs T_w(k) = T_in + Σ dTw[0..k-1].
    dTw_sections = [T_z[k + 1] - T_z[k] for k in range(n)]

    return ChannelOutput(
        velocity=U_mean,
        flow_rate=m_dot / state_out.density,
        friction_factor=cf_mean,
        temp_inlet=T_z[0],
        temp_outlet=T_z[-1],
        temp_rise=T_z[-1] - T_z[0],
        temp_mean=float(np.mean(T_z)),
        heat_coeff=float(np.mean(h_z)),
        heat_coeff_distribution=list(h_z),       # always per-section for gradHZHvar
        temp_rise_distribution=dTw_sections,      # canonical Tw output
        velocity_distribution=list(U_z),          # per-section U_k for feelpp
        density_outlet=state_out.density,
        specific_heat_outlet=state_out.specific_heat,
        converged=converged,
        iterations=iteration + 1,
    )
```

### 2.4 New helper `_solve_mass_flow_axial_var`

This method solves for the mass flow rate `ṁ` such that the section-summed
pressure drop equals `inputs.pressure_drop`.  It mirrors `_solve_velocity` but
operates section-by-section.

```python
def _solve_mass_flow_axial_var(
    self,
    m_dot_guess: float,
    T_z: List[float],
    axial: AxialDiscretization,
    Dh_k: List[float],
    Sh_k: List[float],
    inputs: ThermalHydraulicInput,
    channel_length: float,
) -> tuple[float, List[float]]:
    """
    Solve for mass flow rate ṁ by matching section-summed ΔP to target.

    Returns (m_dot_new, cf_k_list).
    """
    n   = axial.n_sections
    z0  = axial.z_positions[0]
    z_span = axial.z_positions[-1] - z0
    L_k = [axial.z_positions[k + 1] - axial.z_positions[k] for k in range(n)]
    DP_target = inputs.pressure_drop * 1e5 * inputs.extra_pressure_loss  # Pa

    def dp_residual(m_dot):
        dp_total = 0.0
        cf_list  = []
        for k in range(n):
            z_frac  = (axial.z_positions[k] - z0) / z_span
            P_local = inputs.pressure_inlet - inputs.pressure_drop * z_frac
            T_mid   = (T_z[k] + T_z[k + 1]) / 2.0
            state_k = WaterProperties.get_state(T_mid, P_local)
            U_k     = m_dot / (state_k.density * Sh_k[k])
            Re_k    = WaterProperties.compute_reynolds(U_k, Dh_k[k], T_mid, P_local)
            cf_k    = self._friction_model.compute(Re_k, Dh_k[k], 0.055)
            cf_list.append(cf_k)
            dp_total += cf_k * (state_k.density / 2.0) * U_k ** 2 * (L_k[k] / Dh_k[k])
        return dp_total - DP_target, cf_list

    # Simple secant iteration (2–4 steps is usually enough).
    m1 = m_dot_guess
    m2 = m1 * 1.05
    f1, cf_k = dp_residual(m1)
    f2, _    = dp_residual(m2)

    for _ in range(inputs.max_iterations):
        if abs(f2 - f1) < 1e-12:
            break
        m3 = m2 - f2 * (m2 - m1) / (f2 - f1)
        m3 = max(m3, 1e-6)   # guard against negative mass flow
        f3, cf_k = dp_residual(m3)
        if abs(f3) < 1.0:    # 1 Pa tolerance
            m2, f2 = m3, f3
            break
        m1, f1 = m2, f2
        m2, f2 = m3, f3

    return m2, cf_k
```

---

## 3. Changes to `feelpp.py`

### 3.1 Extracting per-section geometry from FeelPP parameters

`_build_input_from_feelpp` must read per-section `Dh` and `Sh` arrays when
`cooling_level == CoolingLevel.GRAD_HZH_VAR`.

The FeelPP parameter convention is:

| p_params key  | type                    | meaning                                  |
|---------------|-------------------------|------------------------------------------|
| `"DhHZ"`      | `List[List[str]]`       | `[channel_idx][section_idx]` param names |
| `"ShHZ"`      | `List[List[str]]`       | same for cross-section                   |

Add a new branch inside the per-channel loop:

```python
axial_disc = None
if cooling_level.is_axial and isinstance(TwH[i], dict):
    axial_disc = self._extract_axial_discretization(
        i, cname, TwH[i], dict_df[target].get("FluxZ"), basedir,
        # variable geometry lists (None for gradHZ / gradHZH):
        dh_params=p_params.get("DhHZ", []),
        sh_params=p_params.get("ShHZ", []),
        parameters=parameters,
        channel_idx=i,
        cooling_level=cooling_level,
    )
```

### 3.2 Updated `_extract_axial_discretization` signature

```python
def _extract_axial_discretization(
    self,
    channel_idx,
    channel_name,
    Tw_dict,
    FluxZ_df,
    basedir,
    *,
    dh_params=None,
    sh_params=None,
    parameters=None,
    cooling_level=None,
) -> Optional[AxialDiscretization]:
```

At the end of the method, before constructing `AxialDiscretization`, add:

```python
hydraulic_diameters = None
cross_sections_list = None

if (
    cooling_level is not None
    and cooling_level.has_variable_geometry
    and dh_params
    and sh_params
    and channel_idx < len(dh_params)
):
    hydraulic_diameters = [parameters[p] for p in dh_params[channel_idx]]
    cross_sections_list = [parameters[p] for p in sh_params[channel_idx]]

return AxialDiscretization(
    z_positions=z_positions,
    power_distribution=power_distribution,
    hydraulic_diameters=hydraulic_diameters,
    cross_sections=cross_sections_list,
)
```

### 3.3 `_extract_parameter_updates` — velocity distribution

Add handling for `velocity_distribution` under `gradHZHvar`:

```python
# gradHZHvar: per-section velocity profile.
UwHZ_params = p_params.get("UwHZ", [])
if UwHZ_params and channel_out.velocity_distribution is not None:
    ch_idx = list(th_output.channels).index(channel_out)
    if ch_idx < len(UwHZ_params):
        for param_name, U_k in zip(UwHZ_params[ch_idx], channel_out.velocity_distribution):
            updates[param_name] = U_k
```

### 3.4 `_update_dict_df` — velocity distribution

```python
# gradHZHvar: per-section velocity profile.
if channel_out.velocity_distribution is not None:
    dict_df[target].setdefault("UwZ", {})[f"UwZ_{cname}"] = (
        channel_out.velocity_distribution
    )
```

---

## 4. Example `p_params` structure for `gradHZHvar`

```python
p_params = {
    # Per-channel scalar geometry (channel mean, used by geometry object).
    "Dh":    ["Dh_H1", "Dh_H2"],
    "Sh":    ["Sh_H1", "Sh_H2"],

    # Per-channel axial boundary conditions.
    "TwH":   ["TwH_H1", "TwH_H2"],   # or dicts with "filename"
    "dTwH":  ["dTwH_H1", "dTwH_H2"],
    "hwH":   ["hwH_H1", "hwH_H2"],
    "ZmaxH": ["Zmax_H1", "Zmax_H2"],
    "ZminH": ["Zmin_H1", "Zmin_H2"],

    # Per-section variable geometry (gradHZHvar only).
    # Shape: [n_channels][n_sections].
    "DhHZ": [
        ["DhZ0_H1", "DhZ1_H1", "DhZ2_H1"],   # channel 0
        ["DhZ0_H2", "DhZ1_H2", "DhZ2_H2"],   # channel 1
    ],
    "ShHZ": [
        ["ShZ0_H1", "ShZ1_H1", "ShZ2_H1"],
        ["ShZ0_H2", "ShZ1_H2", "ShZ2_H2"],
    ],

    # Output parameters — names for per-section results.
    "dTwHZ": [
        ["dTwZ0_H1", "dTwZ1_H1", "dTwZ2_H1"],
        ["dTwZ0_H2", "dTwZ1_H2", "dTwZ2_H2"],
    ],
    "hwHZ": [
        ["hwZ0_H1", "hwZ1_H1", "hwZ2_H1"],
        ["hwZ0_H2", "hwZ1_H2", "hwZ2_H2"],
    ],
    "UwHZ": [
        ["UwZ0_H1", "UwZ1_H1", "UwZ2_H1"],   # optional
        ["UwZ0_H2", "UwZ1_H2", "UwZ2_H2"],
    ],
}
```

---

## 5. Testing requirements

### Unit: `_compute_channel_axial_var`

- **Constant geometry degenerates to `_compute_channel_axial`**: when all
  `Dh_k` and `Sh_k` are equal, the results of `_compute_channel_axial_var`
  should match `_compute_channel_axial` within tolerance.
- **Mass conservation**: verify `ṁ = ρ_k · U_k · Sh_k` is constant across
  sections (within floating-point tolerance).
- **Pressure drop matching**: the sum of section pressure drops equals
  `inputs.pressure_drop` (within solver tolerance).
- **Tw output shape**: `len(temp_rise_distribution) == n_sections`.
- **Tw output sum**: `sum(temp_rise_distribution) ≈ temp_rise`.
- **dTw reconstruction**: `T_in + cumsum(dTw)[k] ≈ T_z[k+1]` for all k.

### Unit: `AxialDiscretization` validation

- Supplying only `hydraulic_diameters` (without `cross_sections`) raises
  `ValueError`.
- Length mismatch (n_sections ≠ len(hydraulic_diameters)) raises `ValueError`.

### Integration: `feelpp.py`

- `_extract_axial_discretization` correctly populates `hydraulic_diameters` and
  `cross_sections` from `DhHZ` / `ShHZ` p_params.
- `_extract_parameter_updates` writes per-section `UwHZ` values when provided.
- `_update_dict_df` stores `UwZ` in `dict_df`.

### Regression

- All existing tests for `gradHZ` and `gradHZH` still pass unchanged.

---

## 6. Implementation checklist

- [ ] `channel.py`: add `GRAD_HZH_VAR`, update `has_per_section_h`,
      add `has_variable_geometry`, extend `AxialDiscretization`,
      add `velocity_distribution` to `ChannelOutput`.
- [ ] `thermohydraulics.py`: validation in `_validate_inputs`, routing in
      `compute()`, new `_compute_channel_axial_var`, new
      `_solve_mass_flow_axial_var`.
- [ ] `feelpp.py`: updated `_extract_axial_discretization` signature,
      `DhHZ`/`ShHZ` extraction, `UwHZ` output in `_extract_parameter_updates`
      and `_update_dict_df`.
- [ ] Tests: constant-geometry regression, mass conservation, pressure balance,
      feelpp extraction round-trip.
- [ ] `README.md`: add `gradHZHvar` to the cooling-level table; add an example
      snippet showing how to construct `AxialDiscretization` with variable
      geometry.

---

## 7. Design constraints (do not change)

- `temp_distribution` (boundary temperature array) is **not** a field of
  `ChannelOutput`; it remains a local variable inside the solver.
- The Tw feelpp output for all axial levels is `temp_rise_distribution` only.
- The `CoolingLevel` string values must stay backward-compatible with the
  `--cooling` CLI argument; `"gradHZHvar"` is the new value.
- `_compute_channel_axial` is left **unchanged**; variable-geometry logic lives
  exclusively in `_compute_channel_axial_var`.
