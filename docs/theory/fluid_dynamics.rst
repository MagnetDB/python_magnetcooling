Fluid Dynamics Theory
=====================

Friction Factors
----------------

Darcy Friction Factor
~~~~~~~~~~~~~~~~~~~~~

The Darcy friction factor relates pressure drop to flow velocity in pipes:

.. math::

   \Delta P = f \frac{L}{D} \frac{\rho v^2}{2}

where:
* :math:`\Delta P` is pressure drop
* :math:`f` is the Darcy friction factor
* :math:`L` is pipe length
* :math:`D` is pipe diameter
* :math:`\rho` is fluid density
* :math:`v` is flow velocity

Laminar Flow
~~~~~~~~~~~~

For laminar flow (:math:`Re < 2300`):

.. math::

   f = \frac{64}{Re}

Turbulent Flow - Colebrook Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For turbulent flow in rough pipes:

.. math::

   \frac{1}{\sqrt{f}} = -2.0 \log_{10}\left(\frac{\epsilon/D}{3.7} + \frac{2.51}{Re\sqrt{f}}\right)

where :math:`\epsilon` is the pipe roughness.

Reynolds Number
---------------

The Reynolds number characterizes flow regime:

.. math::

   Re = \frac{\rho v D}{\mu} = \frac{v D}{\nu}

where:
* :math:`\mu` is dynamic viscosity
* :math:`\nu` is kinematic viscosity

Flow Regimes
~~~~~~~~~~~~

* Laminar: :math:`Re < 2300`
* Transitional: :math:`2300 < Re < 4000`
* Turbulent: :math:`Re > 4000`

References
----------

Coming soon.
