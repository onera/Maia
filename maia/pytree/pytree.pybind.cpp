#include <pybind11/pybind11.h>

#include "maia/pytree/pytree.pybind.hpp"

#include "maia/pytree/cgns_keywords/cgns_keywords.pybind.hpp"


namespace py = pybind11;

void register_pytree_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("pytree");

  m.doc() = "pybind11 pytree module"; // optional module docstring


  register_cgns_keywords_module(m);
  register_cgns_names_module(m);
  
}
