#!/bin/bash

if [ -z "$MAIA_SOURCE_DIR" ]; then
  echo "variable MAIA_SOURCE_DIR not set, you need to define it"
  exit 1
fi

# echo each command of the script with bold green "+ " in front of it
# See https://stackoverflow.com/q/26067916/1583122
COLOR_GREEN="\[\033[38;2;77;175;74m\]"
COLOR_RESET="\[\033[0m\]"
COLOR_BOLD="\[\033[1m\]"
PS4="${COLOR_BOLD}${COLOR_GREEN}${PS4}${COLOR_RESET}"
set -x

# fail on first non-0 returning command (warning: with this, do not use `if` below)
set -e

cd $MAIA_SOURCE_DIR
mkdir -p build/alpine_int64
cd build/alpine_int64

# Note: the software stack used by Maia is already loaded on the image (PYTHONPATH...)
cmake -S $MAIA_SOURCE_DIR
make -j 24
source source.sh
mpirun -np 4 test/maia_doctest_unit_tests
mpirun -np 4 python3 -m pytest $MAIA_SOURCE_DIR/maia/
mpirun -np 8 python3 -m pytest $MAIA_SOURCE_DIR/test --scheduler=dynamic
