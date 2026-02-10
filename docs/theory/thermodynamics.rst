Thermodynamics
==============

Water Properties
----------------

IAPWS Standard
~~~~~~~~~~~~~~

The package uses the International Association for the Properties of Water and Steam (IAPWS)
formulations for accurate water property calculations.

Key properties calculated:

* Density (:math:`\rho`)
* Dynamic viscosity (:math:`\mu`)
* Thermal conductivity (:math:`k`)
* Specific heat capacity (:math:`c_p`)
* Prandtl number (:math:`Pr`)

Prandtl Number
~~~~~~~~~~~~~~

The Prandtl number relates momentum diffusivity to thermal diffusivity:

.. math::

   Pr = \frac{\mu c_p}{k} = \frac{\nu}{\alpha}

where :math:`\alpha` is thermal diffusivity.

Energy Balance
--------------

For a control volume with heat input:

.. math::

   Q = \dot{m} c_p \Delta T

where:
* :math:`Q` is heat transfer rate
* :math:`\dot{m}` is mass flow rate
* :math:`c_p` is specific heat capacity
* :math:`\Delta T` is temperature change

Heat Flux
~~~~~~~~~

Heat flux at a surface:

.. math::

   q'' = h(T_w - T_b)

where:
* :math:`q''` is heat flux
* :math:`h` is heat transfer coefficient
* :math:`T_w` is wall temperature
* :math:`T_b` is bulk fluid temperature

References
----------

Coming soon.
