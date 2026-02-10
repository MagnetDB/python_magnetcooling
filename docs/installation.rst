Installation
============

Requirements
------------

* Python >= 3.11
* numpy >= 2.0.0
* scipy >= 1.14.0
* pandas >= 2.2.0
* iapws >= 1.4.0
* pint >= 0.17.1
* ht >= 1.2.0

Installation from PyPI
----------------------

.. code-block:: bash

   pip install python_magnetcooling

Development Installation
------------------------

To install the development version with all optional dependencies:

.. code-block:: bash

   git clone https://github.com/MagnetDB/python_magnetcooling
   cd python_magnetcooling
   pip install -e ".[dev,docs,viz]"

Optional Dependencies
---------------------

Development tools
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install python_magnetcooling[dev]

This includes:

* pytest - Testing framework
* pytest-cov - Test coverage
* black - Code formatter
* mypy - Type checker
* flake8 - Linter

Documentation
~~~~~~~~~~~~~

.. code-block:: bash

   pip install python_magnetcooling[docs]

This includes:

* sphinx - Documentation generator
* sphinx-rtd-theme - Read the Docs theme
* sphinx-autodoc-typehints - Type hints support

Visualization
~~~~~~~~~~~~~

.. code-block:: bash

   pip install python_magnetcooling[viz]

This includes:

* matplotlib - Plotting library
* seaborn - Statistical visualization
