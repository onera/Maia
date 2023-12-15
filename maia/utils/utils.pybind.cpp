#include "maia/utils/utils.pybind.hpp"

#include "maia/utils/ndarray/subset_sum.pybind.hpp"
#include "maia/utils/ndarray/find_duplicate_elt.pybind.hpp"
#include "maia/utils/ndarray/layouts.pybind.hpp"
#include "maia/utils/numbering/numbering.pybind.hpp"
#include "maia/utils/logging/logging.pybind.hpp"

namespace py = pybind11;

void register_utils_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("utils");

  m.doc() = "pybind11 utils module"; // optional module docstring

  register_layouts_module(m);
  register_numbering_module(m);
  register_logging_module(m);

  m.def("search_subset_match", &search_subset_match);

  m.def("is_unique_cst_stride_hash", &is_unique_cst_stride_hash , 
        "Find elements that are duplicated (hash table and solve conflict)");
  m.def("is_unique_cst_stride_sort", &is_unique_cst_stride_sort, 
        "Find elements that are duplicated (hash table and solve conflict with sort algorithm)");
}
