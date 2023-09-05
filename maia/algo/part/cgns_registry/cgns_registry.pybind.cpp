#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/algo/part/cgns_registry/cgns_registry.hpp"

namespace py = pybind11;

cgns_registry make_cgns_registry(const cgns_paths_by_label& paths, py::object mpi4py_obj){
  return cgns_registry(paths, maia::mpi4py_comm_to_comm(mpi4py_obj));
}

void register_cgns_registry_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("cgns_registry");

  m.doc() = "pybind11 utils for cgns_registry plugin"; // optional module docstring

  py::class_<cgns_paths_by_label> (m, "cgns_paths_by_label")
    .def(py::init<>());

  py::bind_vector<std::vector<std::string>>(m, "cgns_paths"); // Neccesary to return into python pybind11 : ticket 2641
  py::bind_vector<std::vector<PDM_g_num_t>>(m, "global_ids");

  py::class_<cgns_registry> (m, "cgns_registry")
    .def(py::init<>(&make_cgns_registry))
    .def("at"          , &cgns_registry::at          , py::return_value_policy::automatic_reference)
    .def("paths"       , &cgns_registry::paths       , py::return_value_policy::automatic_reference)
    .def("global_ids"  , &cgns_registry::global_ids  , py::return_value_policy::automatic_reference)
    .def("distribution", &cgns_registry::distribution, py::return_value_policy::automatic_reference)
    .def("__repr__", [](const cgns_registry& x){
      return to_string(x);
    });

  m.def("add_path",
        py::overload_cast<cgns_paths_by_label&, const std::string&, const std::string&>(add_path),
        "Some doc here");

  m.def("get_global_id_from_path_and_type",
        py::overload_cast<const cgns_registry&, std::string, CGNS::Label>(get_global_id_from_path_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type",
        py::overload_cast<const cgns_registry&, PDM_g_num_t, CGNS::Label>(get_path_from_global_id_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type", [](const cgns_registry& cgns_reg, int g_id, CGNS::Label label) {
    return get_path_from_global_id_and_type(cgns_reg, static_cast<PDM_g_num_t>(g_id), label);
  });

  m.def("get_global_id_from_path_and_type",
        py::overload_cast<const cgns_registry&, std::string, std::string>(get_global_id_from_path_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type",
        py::overload_cast<const cgns_registry&, PDM_g_num_t, std::string>(get_path_from_global_id_and_type),
        "Some doc here");

  m.def("get_path_from_global_id_and_type", [](const cgns_registry& cgns_reg, int g_id, std::string cgns_label_str) {
    return get_path_from_global_id_and_type(cgns_reg, static_cast<PDM_g_num_t>(g_id), cgns_label_str);
  });

}
