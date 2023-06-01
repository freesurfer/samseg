#!/bin/bash -e
# This is an auxiliary script to build and test SAMSEG wheels
PYTHON_PATH=$1
$PYTHON_PATH setup.py bdist_wheel
$PYTHON_PATH -m pip install samseg -f dist/
$PYTHON_PATH -m pip install pytest
$PYTHON_PATH -m pip install tensorflow
$PYTHON_PATH -m pytest samseg/tests
rm samseg/gems/*.so 
