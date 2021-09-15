#include "maia/utils/yaml/parse_yaml_cgns.hpp"


#include "pybind11/embed.h"
namespace py = pybind11;
#include "cpp_cgns/interop/pycgns_converter.hpp"


namespace maia {


auto
to_node(const std::string& yaml_str) -> cgns::tree {
  py::scoped_interpreter guard;

  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");
  py::object py_tree = parse_yaml_cgns.attr("to_node")(yaml_str);
  return cgns::to_cpp_tree_copy(py_tree);
}
auto
to_nodes(const std::string& yaml_str) -> cgns::tree {
  py::scoped_interpreter guard;

  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");
  py::object py_tree = parse_yaml_cgns.attr("to_nodes")(yaml_str);
  
  return cgns::to_cpp_tree_copy(py_tree);
}
auto
to_cgns_tree(const std::string& yaml_str) -> cgns::tree {
  py::scoped_interpreter guard;

  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");
  py::object py_tree = parse_yaml_cgns.attr("to_cgns_tree")(yaml_str);
  
  return cgns::to_cpp_tree_copy(py_tree);
}


} // maia
