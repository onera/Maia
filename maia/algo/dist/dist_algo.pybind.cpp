#include "maia/algo/dist/dist_algo.pybind.hpp"
#if __cplusplus > 201703L
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "maia/algo/dist/elements_to_ngons/elements_to_ngons.hpp"

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

const auto generate_interior_faces_and_parents = apply_cpp_cgns_par_function_to_py_base(maia::generate_interior_faces_and_parents);
const auto elements_to_ngons               = apply_cpp_cgns_par_function_to_py_base(maia::elements_to_ngons);


void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");

  m.def("generate_interior_faces_and_parents"     , generate_interior_faces_and_parents     , "Generate TRI_3_interior and QUAD_4_interior element sections, and adds ParentElement to interior and exterior faces");
  m.def("elements_to_ngons"                       , elements_to_ngons                       , "Convert to NGon");

}
#else //C++==17

namespace py = pybind11;
void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");
  
}
#endif //C++>17
