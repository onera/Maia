#include "maia/algo/tree_algo.pybind.hpp"
#if __cplusplus > 201703L
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/algo/common/poly_algorithm.hpp"

namespace py = pybind11;

template<class F> auto
apply_cpp_cgns_function_to_pytree(F&& f) {
  return [&f](py::list pt) {
    cgns::tree t = cgns::to_cpp_tree(pt);
    f(t);
    update_py_tree(std::move(t),pt);
  };
}

const auto indexed_to_interleaved_connectivity = apply_cpp_cgns_function_to_pytree(maia::indexed_to_interleaved_connectivity);
const auto interleaved_to_indexed_connectivity = apply_cpp_cgns_function_to_pytree(maia::interleaved_to_indexed_connectivity);


void register_tree_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("tree_algo");
  m.def("indexed_to_interleaved_connectivity" , indexed_to_interleaved_connectivity , "Turn NGon or NFace description with ElementStartOffset to old convention");
  m.def("interleaved_to_indexed_connectivity" , interleaved_to_indexed_connectivity , "Turn NGon or NFace description without ElementStartOffset to new convention");

  
}
#else //C++==17

namespace py = pybind11;
void register_tree_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("tree_algo");
  
}
#endif //C++>17
