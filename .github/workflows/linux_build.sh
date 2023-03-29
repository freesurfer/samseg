#!/bin/bash -e
# This is an auxiliary script to build SimNIBS wheels
PYTHON_PATH=$1
$PYTHON_PATH setup.py bdist_wheel
$PYTHON_PATH -m pip install charm-gems -f dist/
$PYTHON_PATH -c 'import charm_gems'
rm charm_gems/*.so gems_python/*.so
