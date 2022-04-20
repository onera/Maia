#pragma once
#include <mpi.h>
#include "mpi4py/mpi4py.MPI.h"
#include <pybind11/pybind11.h>


namespace maia {

namespace py = pybind11;

inline MPI_Comm
mpi4py_comm_to_comm(py::handle mpi4py_obj) {
  return ((PyMPICommObject*)mpi4py_obj.ptr())->ob_mpi;
}

} // maia
