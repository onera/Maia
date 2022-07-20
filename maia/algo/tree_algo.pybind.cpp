#include "maia/algo/tree_algo.pybind.hpp"
#if __cplusplus > 201703L
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/algo/poly_algo/poly_algorithm.hpp"

namespace py = pybind11;

template<class F> auto
apply_cpp_cgns_function_to_py_base(F&& f) {
  return [&f](py::list py_base) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    f(base);
    update_py_tree(std::move(base),py_base);
  };
}
template<class F> auto
apply_cpp_cgns_par_function_to_py_base(F&& f) {
  return [&f](py::list py_base, py::object mpi4py_comm) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);
    f(base,comm);
    update_py_tree(std::move(base),py_base);
  };
}

const auto ngon_new_to_old = apply_cpp_cgns_function_to_py_base(maia::ngon_new_to_old);
const auto ngon_old_to_new = apply_cpp_cgns_function_to_py_base(maia::ngon_old_to_new);


void register_tree_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("tree_algo");
  m.def("ngon_new_to_old" , ngon_new_to_old , "Turn Ngon description with ElementStartOffset to old convention");
  m.def("ngon_old_to_new" , ngon_old_to_new , "Turn Ngon description without ElementStartOffset to new convention");

  
}
#else //C++==17

namespace py = pybind11;
void register_tree_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("tree_algo");
  
}
#endif //C++>17
