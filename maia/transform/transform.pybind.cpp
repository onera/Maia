#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "maia/transform/__old/partition_with_boundary_first/partition_with_boundary_first.hpp"
#include "maia/transform/__old/convert_to_simple_connectivities.hpp"
#include "maia/transform/__old/remove_ghost_info.hpp"
#include "maia/generate/__old/nfaces_from_ngons.hpp"
#include "maia/utils/mpi4py.hpp"
#include "maia/transform/merge_by_elt_type.hpp"
#include "maia/transform/fsdm_distribution.hpp"
#include "maia/transform/gcs_only_for_ghosts.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

auto partition_with_boundary_first(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::partition_with_boundary_first(base,cgns::factory(&alloc),MPI_COMM_WORLD);

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

auto sort_nface_into_simple_connectivities(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::sort_nface_into_simple_connectivities(base,cgns::factory(&alloc));

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

auto convert_to_simple_connectivities(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::convert_to_simple_connectivities(base,cgns::factory(&alloc));

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

auto add_nfaces(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::add_nfaces(base,cgns::factory(&alloc));

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

auto remove_ghost_info(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::remove_ghost_info(base,cgns::factory(&alloc),MPI_COMM_WORLD);

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

auto merge_by_elt_type(py::list py_base, py::object mpi4py_comm) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);
  MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::merge_by_elt_type(base,cgns::factory(&alloc),comm);

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}
auto add_fsdm_distribution(py::list py_base, py::object mpi4py_comm) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);
  MPI_Comm comm = maia::mpi4py_comm_to_comm(mpi4py_comm);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::add_fsdm_distribution(base,cgns::factory(&alloc),comm);

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}
auto gcs_only_for_ghosts(py::list py_base) -> void {
  cgns::tree base = cgns::to_cpp_tree(py_base);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::gcs_only_for_ghosts(base,cgns::factory(&alloc));

  update_and_transfer_ownership_to_py_tree(base,alloc,py_base);
}

PYBIND11_MODULE(transform, m) {
  m.doc() = "C++ maia functions wrapped by pybind";

  m.def("partition_with_boundary_first"        , &partition_with_boundary_first        , "ngon sorted with boundary faces first");
  m.def("sort_nface_into_simple_connectivities", &sort_nface_into_simple_connectivities, "sort nface into tet, prism, pyra, hex");
  m.def("convert_to_simple_connectivities"     , &convert_to_simple_connectivities     , "turn ngon with boundary first and nface to tri, quad, tet, prism, pyra, hex");
  m.def("add_nfaces"                           , &add_nfaces                           , "add nface Elements_t from ngons with ParentElements");
  m.def("remove_ghost_info"                    , &remove_ghost_info                    , "Remove ghost nodes and ghost elements of base");
  m.def("merge_by_elt_type"                    , &merge_by_elt_type                    , "For a distributed base, merge Elements_t nodes the same type and does the associated renumbering");
  m.def("add_fsdm_distribution"                , &add_fsdm_distribution                , "Add FSDM-specific distribution info");
  m.def("gcs_only_for_ghosts"                  , &gcs_only_for_ghosts                  , "For GridConnectivities, keep only in the PointList the ones that are ghosts");
}
