#include <iostream>
#include <vector>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "maia/utils/mpi4py.hpp"
#include "maia/cgns_registry/cgns_registry.hpp"

namespace py = pybind11;

cgns_registry make_cgns_registry(const cgns_paths_by_label& paths, py::object mpi4py_obj){
  return cgns_registry(paths, mpi4py_comm_to_comm(mpi4py_obj));
}


PYBIND11_MODULE(cgns_registry, m) {
  m.doc() = "pybind11 utils for cgns_registry plugin"; // optional module docstring

  py::class_<cgns_paths_by_label> (m, "cgns_paths_by_label")
    .def(py::init<>());

  m.def("make_cgns_registry", &make_cgns_registry,
        "Some doc here");

  py::class_<cgns_registry> (m, "cgns_registry")
    .def(py::init<>(&make_cgns_registry))
    .def("__repr__", [](const cgns_registry& x){
      return to_string(x);
    });

  m.def("add_path",
        py::overload_cast<cgns_paths_by_label&, cgns_path, CGNS::Label::kind>(add_path),
        "Some doc here");

  m.def("add_path",
        py::overload_cast<cgns_paths_by_label&, cgns_path, const std::string&>(add_path),
        "Some doc here");


  m.def("get_global_id_from_path_and_type",
        py::overload_cast<const cgns_registry&, std::string, CGNS::Label::kind>(get_global_id_from_path_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type",
        py::overload_cast<const cgns_registry&, int, CGNS::Label::kind>(get_path_from_global_id_and_type),
        "Some doc here");

  m.def("get_global_id_from_path_and_type",
        py::overload_cast<const cgns_registry&, std::string, std::string>(get_global_id_from_path_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type",
        py::overload_cast<const cgns_registry&, int, std::string>(get_path_from_global_id_and_type),
        "Some doc here");


}
