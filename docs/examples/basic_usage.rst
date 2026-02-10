Basic Usage Examples
====================

Water Properties
----------------

Calculate various water properties at different conditions:

.. code-block:: python

   from python_magnetcooling import water_properties
   
   # Example 1: Properties at ambient conditions
   T = 293.15  # K (20°C)
   P = 101325  # Pa (1 atm)
   
   props = water_properties.calculate_properties(T, P)
   print(f"Density: {props['rho']:.2f} kg/m³")
   print(f"Viscosity: {props['mu']:.6e} Pa·s")
   print(f"Thermal conductivity: {props['k']:.4f} W/(m·K)")
   print(f"Specific heat: {props['cp']:.2f} J/(kg·K)")
   
   # Example 2: High pressure conditions
   T = 350  # K
   P = 2e6  # Pa (20 bar)
   
   props = water_properties.calculate_properties(T, P)
   print(f"\nAt high pressure:")
   print(f"Density: {props['rho']:.2f} kg/m³")
   print(f"Prandtl number: {props['Pr']:.4f}")

Heat Transfer Calculations
---------------------------

Using different correlations for heat transfer:

.. code-block:: python

   from python_magnetcooling import correlations
   
   # Turbulent flow in a pipe
   Re = 50000   # Reynolds number
   Pr = 5.0     # Prandtl number
   
   # Gnielinski correlation
   Nu_g = correlations.gnielinski(Re, Pr)
   print(f"Gnielinski Nu: {Nu_g:.2f}")
   
   # Dittus-Boelter correlation
   Nu_db = correlations.dittus_boelter(Re, Pr)
   print(f"Dittus-Boelter Nu: {Nu_db:.2f}")

Friction Factor Calculations
-----------------------------

Calculate friction factors for various flow conditions:

.. code-block:: python

   from python_magnetcooling import friction
   
   # Smooth pipe
   Re = 10000
   epsilon = 0.0  # Smooth pipe
   
   f_smooth = friction.colebrook(Re, epsilon)
   print(f"Smooth pipe friction factor: {f_smooth:.6f}")
   
   # Rough pipe
   epsilon = 0.001  # Relative roughness
   f_rough = friction.colebrook(Re, epsilon)
   print(f"Rough pipe friction factor: {f_rough:.6f}")
   
   # Laminar flow
   Re_lam = 1000
   f_lam = friction.laminar(Re_lam)
   print(f"Laminar flow friction factor: {f_lam:.6f}")
