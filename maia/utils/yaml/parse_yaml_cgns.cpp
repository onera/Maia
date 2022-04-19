#if __cplusplus > 201703L
#include "maia/utils/yaml/parse_yaml_cgns.hpp"


#include "pybind11/embed.h"
namespace py = pybind11;
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "std_e/utils/embed_python.hpp"
#include "std_e/future/ranges.hpp"


namespace maia {


auto
to_node(const std::string& yaml_str) -> cgns::tree {
  std_e::throw_if_no_python_interpreter(__func__);
  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");
  py::object py_tree = parse_yaml_cgns.attr("to_node")(yaml_str);
  return cgns::to_cpp_tree_copy(py_tree);
}
auto
to_nodes(const std::string& yaml_str) -> std::vector<cgns::tree> {
  std_e::throw_if_no_python_interpreter(__func__);
  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");

  #if defined REAL_GCC && __GNUC__ >= 11
    auto py_trees = parse_yaml_cgns.attr("to_nodes")(yaml_str);
    auto to_cpp_tree_copy_fn = [](const py::handle& py_tree) {
      return
        cgns::to_cpp_tree_copy(
          py::reinterpret_borrow<py::list>(py_tree)
        );
    };
    return py_trees | std::views::transform(to_cpp_tree_copy_fn) | std_e::to_vector();
  #else // Note : with the previous code, GCC 10 would call the cgns::tree copy ctor: this is not what we want
    py::list py_trees = parse_yaml_cgns.attr("to_nodes")(yaml_str);
    int n_node = len(py_trees);
    std::vector<cgns::tree> res(n_node);
    for (int i=0; i<n_node; ++i) {
      auto py_tree = py_trees[i];
      auto x = cgns::to_cpp_tree_copy(
        py::reinterpret_borrow<py::list>(py_tree)
      );
      res[i] = std::move(std::move(x));
    }
    return res;
  #endif
}
auto
to_cgns_tree(const std::string& yaml_str) -> cgns::tree {
  std_e::throw_if_no_python_interpreter(__func__);
  auto parse_yaml_cgns = py::module_::import("maia.utils.parse_yaml_cgns");
  py::object py_tree = parse_yaml_cgns.attr("to_cgns_tree")(yaml_str);

  return cgns::to_cpp_tree_copy(py_tree);
}


} // maia
#endif // C++>17
