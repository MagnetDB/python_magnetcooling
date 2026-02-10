Quick Start
===========

Basic Usage
-----------

Water Properties
~~~~~~~~~~~~~~~~

Calculate water properties at given temperature and pressure:

.. code-block:: python

   from python_magnetcooling import water_properties
   
   # Temperature in Kelvin, Pressure in Pascal
   T = 300  # K
   P = 1e5  # Pa
   
   props = water_properties.calculate_properties(T, P)
   print(f"Density: {props['rho']:.2f} kg/m³")
   print(f"Viscosity: {props['mu']:.6e} Pa·s")
   print(f"Thermal conductivity: {props['k']:.4f} W/(m·K)")

Heat Transfer Correlations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use correlations for heat transfer coefficient calculations:

.. code-block:: python

   from python_magnetcooling import correlations
   
   # Calculate Nusselt number for turbulent flow
   Re = 10000  # Reynolds number
   Pr = 5.0    # Prandtl number
   
   Nu = correlations.gnielinski(Re, Pr)
   print(f"Nusselt number: {Nu:.2f}")

Friction Factor
~~~~~~~~~~~~~~~

Calculate friction factors for pipe flow:

.. code-block:: python

   from python_magnetcooling import friction
   
   Re = 10000  # Reynolds number
   epsilon = 0.0001  # Relative roughness
   
   f = friction.colebrook(Re, epsilon)
   print(f"Friction factor: {f:.6f}")

Channel Analysis
~~~~~~~~~~~~~~~~

Perform thermal-hydraulic analysis of a cooling channel:

.. code-block:: python

   from python_magnetcooling import channel
   
   # Define channel geometry
   D = 0.01  # Diameter in meters
   L = 1.0   # Length in meters
   
   # Flow conditions
   mdot = 0.1  # Mass flow rate in kg/s
   T_in = 293  # Inlet temperature in K
   P_in = 2e5  # Inlet pressure in Pa
   q = 1e5     # Heat flux in W/m²
   
   # Analyze channel
   result = channel.analyze(D, L, mdot, T_in, P_in, q)
   print(f"Outlet temperature: {result['T_out']:.2f} K")
   print(f"Pressure drop: {result['dP']:.2f} Pa")

Next Steps
----------

* Check the :doc:`api/index` for detailed module documentation
* See :doc:`examples/index` for more complete examples
* Read :doc:`theory/index` for theoretical background
