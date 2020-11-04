#pragma once
#include <mpi.h>
#include "mpi4py/mpi4py.MPI.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

inline MPI_Comm&
mpi4py_comm_to_comm(py::object& mpi4py_obj) {
  return (MPI_Comm&)(((PyMPICommObject*) mpi4py_obj.ptr())->ob_mpi);
}
