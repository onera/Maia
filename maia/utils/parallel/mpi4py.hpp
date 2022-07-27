#pragma once
#include <mpi.h>
#include "mpi4py/mpi4py.MPI.h"
#include <pybind11/pybind11.h>


namespace maia {

namespace py = pybind11;

inline auto
mpi4py_comm_to_comm(py::handle mpi4py_comm) -> MPI_Comm {
  return ((PyMPICommObject*)mpi4py_comm.ptr())->ob_mpi;
}
inline auto
comm_to_mpi4py_comm(MPI_Comm comm) -> py::object {
  auto mpi4py_mod = py::module_::import("mpi4py.MPI");
  py::object mpi4py_comm = mpi4py_mod.attr("Comm")();
  ((PyMPICommObject*)mpi4py_comm.ptr())->ob_mpi = comm;
  return mpi4py_comm;
}

} // maia
