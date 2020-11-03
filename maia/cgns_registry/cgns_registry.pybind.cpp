#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "maia/cgns_registry/cgns_registry.hpp"
#include "mpi4py/mpi4py.MPI.h"

namespace py = pybind11;


MPI_Comm& hello_mpi4py(py::object mpi4py_obj){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return (MPI_Comm&)(((PyMPICommObject*) mpi4py_obj.ptr())->ob_mpi);
}

PYBIND11_MODULE(cgns_registry, m) {
  m.doc() = "pybind11 utils for cgns_registry plugin"; // optional module docstring

  py::class_<cgns_paths_by_label> (m, "cgns_paths_by_label")
    .def(py::init<>());

  py::class_<MPI_Comm> (m, "MPI_Comm")
    .def(py::init<>());

  m.def("hello_mpi4py", &hello_mpi4py,
        "Some doc here");

  py::class_<cgns_registry> (m, "cgns_registry")
    .def(py::init<const cgns_paths_by_label&, MPI_Comm&>());
  // m.def("add_path", add_path,
  //       py::arg("paths"),
  //       py::arg("path"),
  //       py::arg("label"));
  m.def("add_path", py::overload_cast<cgns_paths_by_label&, cgns_path, CGNS::Label::kind>(add_path),
        "Some doc here");
  m.def("add_path", py::overload_cast<cgns_paths_by_label&, cgns_path, const std::string&>(add_path),
        "Some doc here");
}
