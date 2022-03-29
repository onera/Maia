#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "maia/transform/__old/put_boundary_first/put_boundary_first.hpp"
#include "maia/transform/__old/convert_to_std_elements.hpp"
#include "maia/transform/__old/remove_ghost_info.hpp"
#include "maia/generate/__old/nfaces_from_ngons.hpp"
#include "maia/utils/mpi4py.hpp"
#include "maia/transform/merge_by_elt_type.hpp"
#include "maia/transform/fsdm_distribution.hpp"
#include "maia/transform/gcs_only_for_ghosts.hpp"
#include "maia/transform/split_boundary_subzones_according_to_bcs.hpp"
#include "maia/transform/poly_algorithm.hpp"
#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"
#include "maia/connectivity/std_elements_to_ngons.hpp"

#include <pybind11/pybind11.h>

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

const auto put_boundary_first = apply_cpp_cgns_par_function_to_py_base(maia::put_boundary_first);
const auto remove_ghost_info             = apply_cpp_cgns_par_function_to_py_base(maia::remove_ghost_info);
const auto merge_by_elt_type             = apply_cpp_cgns_par_function_to_py_base(maia::merge_by_elt_type);
const auto add_fsdm_distribution         = apply_cpp_cgns_par_function_to_py_base(maia::add_fsdm_distribution);
const auto split_boundary_subzones_according_to_bcs = apply_cpp_cgns_par_function_to_py_base(maia::split_boundary_subzones_according_to_bcs);
const auto generate_interior_faces_and_parents      = apply_cpp_cgns_par_function_to_py_base(maia::generate_interior_faces_and_parents); // TODO move
const auto std_elements_to_ngons                    = apply_cpp_cgns_par_function_to_py_base(maia::std_elements_to_ngons); // TODO move

const auto convert_zone_to_std_elements = apply_cpp_cgns_function_to_py_base(maia::convert_zone_to_std_elements);
const auto gcs_only_for_ghosts          = apply_cpp_cgns_function_to_py_base(cgns::gcs_only_for_ghosts);
const auto ngon_new_to_old              = apply_cpp_cgns_function_to_py_base(maia::ngon_new_to_old);



PYBIND11_MODULE(transform, m) {
  m.doc() = "C++ maia functions wrapped by pybind";

  m.def("put_boundary_first"                      , put_boundary_first                      , "ngon sorted with boundary faces first");
  m.def("convert_zone_to_std_elements"            , convert_zone_to_std_elements            , "ngon to elements");
  m.def("remove_ghost_info"                       , remove_ghost_info                       , "Remove ghost nodes and ghost elements of base");
  m.def("merge_by_elt_type"                       , merge_by_elt_type                       , "For a distributed base, merge Elements_t nodes the same type and does the associated renumbering");
  m.def("add_fsdm_distribution"                   , add_fsdm_distribution                   , "Add FSDM-specific distribution info");
  m.def("gcs_only_for_ghosts"                     , gcs_only_for_ghosts                     , "For GridConnectivities, keep only in the PointList the ones that are ghosts");
  m.def("split_boundary_subzones_according_to_bcs", split_boundary_subzones_according_to_bcs, "Split a ZoneSubRegion node with a PointRange spaning all boundary faces into multiple ZoneSubRegion with a BCRegionName");
  m.def("ngon_new_to_old"                         , ngon_new_to_old                         , "Turn Ngon description with ElementStartOffset to old convension");
  m.def("generate_interior_faces_and_parents"     , generate_interior_faces_and_parents     , "Generate TRI_3_interior and QUAD_4_interior element sections, and adds ParentElement to interior and exterior faces");
  m.def("std_elements_to_ngons"                   , std_elements_to_ngons                   , "Convert to NGon");
}
