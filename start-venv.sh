#! /bin/bash

set +x

# pwd shall contains requirements.txt

VENVDIR=./venv
USE_SYSTEM_PACKAGES=1

if [ ! -d $VENVDIR ]; then
   echo "create Python Virtualenv: VENVDIR=${VENVDIR}"
   if [ "$USE_SYSTEM_PACKAGES" == "1" ]; then
      python -m venv --system-site-packages $VENVDIR
   else
      python -m venv $VENVDIR
   fi
   . $VENVDIR/bin/activate
   python -m pip install -e ".[dev]"
   deactivate
fi

# add option to properly quit gmsh-env using deactivate
