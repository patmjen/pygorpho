#!/bin/bash

rm -f dist/* &&

module load cuda/9.2 &&
module load gcc/5.2.0 &&

module load python3/3.5.4 &&
source py35/bin/activate &&
python setup.py bdist_wheel &&
deactivate &&

module swap python3/3.6.2 &&
source py36/bin/activate &&
python setup.py bdist_wheel &&
deactivate &&

module swap python3/3.7.5 &&
source py37/bin/activate &&
python setup.py bdist_wheel &&
deactivate &&

module swap python3/3.8.0 &&
source py38/bin/activate &&
python setup.py bdist_wheel &&
deactivate &&

rename linux manylinux1 dist/* &&
python3 -m twine upload dist/*
