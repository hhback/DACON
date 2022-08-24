#!/bin/bash

# PATH Setting
script="`readlink -f "${BASH_SOURCE[0]}"`"
HOMEDIR="`dirname "$script"`"

python ${HOMEDIR}/preprocess.py
python ${HOMEDIR}/train.py
