#!/bin/bash -e
# This is an auxiliary script to build SAMSEG wheels
PYTHON_PATH=$1
$PYTHON_PATH setup.py bdist_wheel
$PYTHON_PATH -m pip install samseg -f dist/
$PYTHON_PATH -c 'import samseg'
rm samseg/gems/*.so 
