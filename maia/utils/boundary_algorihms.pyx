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

# cimport    mpi4py.MPI        as MPI

# MANDATORY :
# ---------
NPY.import_array()

# Include file :
# ------------

# -----------------------------------------------------------------
ctypedef fused integer_kind:
  NPY.int32_t
  NPY.int64_t

# -----------------------------------------------------------------
def convert_to_unstructured():
  """
  """
  # ************************************************************************
  # > Declaration
  # ************************************************************************
  print("convert_to_ngon")

# -----------------------------------------------------------------
def automatic_dispatch(NPY.ndarray[integer_kind, ndim=1, mode='fortran'] face_vtx,
                       NPY.ndarray[NPY.int32_t , ndim=1, mode='fortran'] face_vtx_idx):
  """
  """
  print("automatic_dispatch face_vtx     ---> ", type(face_vtx)    )
  print("automatic_dispatch face_vtx_idx ---> ", type(face_vtx_idx))

