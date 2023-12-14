#include "maia/algo/dist/dist_algo.pybind.hpp"
#if __cplusplus > 201703L
#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/utils/parallel/mpi4py.hpp"
#include "maia/__old/transform/convert_to_std_elements.hpp"
#include "maia/algo/dist/rearrange_element_sections/rearrange_element_sections.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "maia/algo/dist/elements_to_ngons/elements_to_ngons.hpp"
#include "maia/algo/dist/fsdm_distribution/fsdm_distribution.hpp"
#include "maia/__old/transform/put_boundary_first/put_boundary_first.hpp"
#include "maia/algo/dist/split_boundary_subzones_according_to_bcs/split_boundary_subzones_according_to_bcs.hpp"

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
const auto convert_zone_to_std_elements = apply_cpp_cgns_function_to_py_base(maia::convert_zone_to_std_elements);
const auto add_fsdm_distribution         = apply_cpp_cgns_par_function_to_py_base(maia::add_fsdm_distribution);
const auto rearrange_element_sections             = apply_cpp_cgns_par_function_to_py_base(maia::rearrange_element_sections);
const auto put_boundary_first = apply_cpp_cgns_par_function_to_py_base(maia::put_boundary_first);
const auto split_boundary_subzones_according_to_bcs = apply_cpp_cgns_par_function_to_py_base(maia::split_boundary_subzones_according_to_bcs);


void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");

  m.def("generate_interior_faces_and_parents"     , generate_interior_faces_and_parents     , "Generate TRI_3_interior and QUAD_4_interior element sections, and adds ParentElement to interior and exterior faces");
  m.def("elements_to_ngons"                       , elements_to_ngons                       , "Convert to NGon");
  m.def("convert_zone_to_std_elements"            , convert_zone_to_std_elements            , "ngon to elements");
  m.def("add_fsdm_distribution"                   , add_fsdm_distribution                   , "Add FSDM-specific distribution info");
  m.def("rearrange_element_sections"              , rearrange_element_sections              , "For a distributed base, merge Elements_t nodes of the same type and does the associated renumbering");
  m.def("put_boundary_first"                      , put_boundary_first                      , "ngon sorted with boundary faces first");
  m.def("split_boundary_subzones_according_to_bcs", split_boundary_subzones_according_to_bcs, "Split a ZoneSubRegion node with a PointRange spaning all boundary faces into multiple ZoneSubRegion with a BCRegionName");
  
}
#else //C++==17

namespace py = pybind11;
void register_dist_algo_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("dist_algo");
  
}
#endif //C++>17
