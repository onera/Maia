#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "maia/transform/__old/partition_with_boundary_first/partition_with_boundary_first.hpp"
#include "maia/transform/__old/convert_to_std_elements.hpp"
#include "maia/transform/__old/remove_ghost_info.hpp"
#include "maia/generate/__old/nfaces_from_ngons.hpp"
#include "maia/utils/mpi4py.hpp"
#include "maia/transform/merge_by_elt_type.hpp"
#include "maia/transform/fsdm_distribution.hpp"
#include "maia/transform/gcs_only_for_ghosts.hpp"
#include "maia/transform/split_boundary_subzones_according_to_bcs.hpp"
#include "maia/transform/poly_algorithm.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;


template<class F> auto
apply_cpp_cgns_function_to_py_base(F&& f) {
  return [&f](py::list py_base) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    f(base);
    update_and_transfer_ownership_to_py_tree(base,py_base);
  };
}
template<class F> auto
apply_cpp_cgns_par_function_to_py_base(F&& f) {
  return [&f](py::list py_base, py::object mpi4py_comm) {
    cgns::tree base = cgns::to_cpp_tree(py_base);
    MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);
    f(base,comm);
    update_and_transfer_ownership_to_py_tree(base,py_base);
  };
}

const auto partition_with_boundary_first = apply_cpp_cgns_par_function_to_py_base(cgns::partition_with_boundary_first);
const auto remove_ghost_info             = apply_cpp_cgns_par_function_to_py_base(cgns::remove_ghost_info);
const auto merge_by_elt_type             = apply_cpp_cgns_par_function_to_py_base(maia::merge_by_elt_type);
const auto add_fsdm_distribution         = apply_cpp_cgns_par_function_to_py_base(maia::add_fsdm_distribution);

const auto sort_nfaces_by_element_type              = apply_cpp_cgns_function_to_py_base(cgns::sort_nfaces_by_element_type);
const auto sorted_nfaces_to_std_elements            = apply_cpp_cgns_function_to_py_base(cgns::sorted_nfaces_to_std_elements);
const auto add_nfaces                               = apply_cpp_cgns_function_to_py_base(cgns::add_nfaces);
const auto gcs_only_for_ghosts                      = apply_cpp_cgns_function_to_py_base(cgns::gcs_only_for_ghosts);
const auto split_boundary_subzones_according_to_bcs = apply_cpp_cgns_function_to_py_base(cgns::split_boundary_subzones_according_to_bcs);
const auto ngon_new_to_old                          = apply_cpp_cgns_function_to_py_base(maia::ngon_new_to_old);
const auto sids_conforming_ngon_nface               = apply_cpp_cgns_function_to_py_base(maia::sids_conforming_ngon_nface);


PYBIND11_MODULE(transform, m) {
  m.doc() = "C++ maia functions wrapped by pybind";

  m.def("partition_with_boundary_first"           , partition_with_boundary_first           , "ngon sorted with boundary faces first");
  m.def("sort_nfaces_by_element_type"             , sort_nfaces_by_element_type              , "sort nface into tet, prism, pyra, hex");
  m.def("sorted_nfaces_to_std_elements"           , sorted_nfaces_to_std_elements           , "turn ngon with boundary first and nface to tri, quad, tet, prism, pyra, hex");
  m.def("add_nfaces"                              , add_nfaces                              , "add nface Elements_t from ngons with ParentElements");
  m.def("remove_ghost_info"                       , remove_ghost_info                       , "Remove ghost nodes and ghost elements of base");
  m.def("merge_by_elt_type"                       , merge_by_elt_type                       , "For a distributed base, merge Elements_t nodes the same type and does the associated renumbering");
  m.def("add_fsdm_distribution"                   , add_fsdm_distribution                   , "Add FSDM-specific distribution info");
  m.def("gcs_only_for_ghosts"                     , gcs_only_for_ghosts                     , "For GridConnectivities, keep only in the PointList the ones that are ghosts");
  m.def("split_boundary_subzones_according_to_bcs", split_boundary_subzones_according_to_bcs, "Split a ZoneSubRegion node with a PointRange spaning all boundary faces into multiple ZoneSubRegion with a BCRegionName");
  m.def("ngon_new_to_old"                         , ngon_new_to_old                         , "Turn Ngon description with ElementStartOffset to old convension");
  m.def("sids_conforming_ngon_nface"              , sids_conforming_ngon_nface              , "Only keep ngon and nface");
}
