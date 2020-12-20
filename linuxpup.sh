#!/bin/bash

set -e

PY_MODULE_VERS=("3.5.4" "3.6.2" "3.7.5" "3.8.0")
PY_VENVS=("py35" "py36" "py37" "py38")

rm -f dist/* 

module load cuda/9.2
module load gcc/5.2.0
module load python3 # Load a module so we can use swap after

for ((i = 0; i < ${#PY_VENVS[@]}; ++i)); do
    # Build package
    module swap python3/${PY_MODULE_VERS[i]}
    source ${PY_VENVS[i]}/bin/activate
    python setup.py bdist_wheel

    # Install and run tests
    pip install pygorpho --no-index --find-links=dist/
    pytest
    pip uninstall pygorpho --yes

    deactivate 
done

rename linux manylinux1 dist/*
python3 -m twine upload dist/*
