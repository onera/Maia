#!python
#cython: boundscheck=True
#cython: cdivision=True
#cython: wraparound=True
#cython: profile=True
#cython: embedsignature=True
# See : https://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiler-directives

# Import Section (Python) :
# ------------------------
import logging     as LOG
import numpy       as NPY
import sys         as SYS
import copy        as CPY

# Import Section (Cython) :
# ------------------------
cimport cython
cimport numpy       as      NPY

cimport    mpi4py.MPI        as MPI

# MANDATORY :
# ---------
NPY.import_array()

# Include file :
# ------------

# -----------------------------------------------------------------
def convert_to_ngon():
  """
  """
  # ************************************************************************
  # > Declaration
  # ************************************************************************
  print("convert_to_ngon")
