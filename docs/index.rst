.. python_magnetcooling documentation master file

Welcome to python_magnetcooling's documentation!
================================================

**python_magnetcooling** is a Python library for thermal-hydraulic calculations for water-cooled high-field magnets.

This package provides tools for:

* Water property calculations using IAPWS standards
* Heat transfer correlations for cooling channels
* Friction factor calculations
* Thermal-hydraulic analysis of heat exchangers
* Channel flow simulations

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   theory/index

Installation
============

Install the package using pip:

.. code-block:: bash

   pip install python_magnetcooling

For development installation:

.. code-block:: bash

   git clone https://github.com/MagnetDB/python_magnetcooling
   cd python_magnetcooling
   pip install -e ".[dev,docs]"

Quick Start
===========

Here's a simple example:

.. code-block:: python

   from python_magnetcooling import water_properties
   
   # Calculate water properties at specific conditions
   T = 300  # K
   P = 1e5  # Pa
   
   props = water_properties.calculate_properties(T, P)
   print(f"Density: {props['rho']} kg/m³")
   print(f"Viscosity: {props['mu']} Pa·s")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
