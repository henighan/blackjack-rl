#!/bin/bash

venv='.work'
echo Install Virtualenv
which -s virtualenv || pip install virtualenv

[ -e $venv/bin/python ] || virtualenv $venv
source $venv/bin/activate

# install python3
which -s .work/bin/python3 || virtualenv -p python3 .work

# Install requirements
pip3 install -r requirements.txt --no-cache-dir
