#include "maia/transform/__old/put_boundary_first/put_boundary_first.hpp"

#include "maia/transform/__old/put_boundary_first/boundary_vertices.hpp"
#include "maia/transform/__old/put_boundary_first/boundary_vertices_at_beginning.hpp"
#include "maia/transform/__old/put_boundary_first/boundary_ngons_at_beginning.hpp"

#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "maia/transform/__old/base_renumbering.hpp"
#include "maia/transform/__old/renumber_point_lists.hpp"


using cgns::tree;
using cgns::tree_range;
using cgns::I4;
using cgns::I8;


namespace maia {


// TODO move
template<class T, class I> auto
_permute_node_value(cgns::node_value& x, const std::vector<I>& perm) -> void {
  auto x_span = view_as_span<T>(x);
  std_e::permute(x_span.begin(), perm);
};
template<class I> auto
permute_node_value(cgns::node_value& x, const std::vector<I>& perm) -> void {
  if (x.data_type()=="I4") return _permute_node_value<cgns::I4>(x,perm);
  if (x.data_type()=="I8") return _permute_node_value<cgns::I8>(x,perm);
  if (x.data_type()=="R4") return _permute_node_value<cgns::R4>(x,perm);
  if (x.data_type()=="R8") return _permute_node_value<cgns::R8>(x,perm);
  throw
    cgns::cgns_exception(
        std::string(__func__)
      + ": CGNS node has a value of data type " + x.data_type()
      + " but it should be I4, I8, R4 or R8"
    );
};
// end TODO move


template<class I> auto
permute_boundary_grid_coords_at_beginning(tree& grid_coords, const std::vector<I>& vertex_permutation) -> void {
  STD_E_ASSERT(label(grid_coords)=="GridCoordinates_t");
  auto coords = get_children_by_label(grid_coords,"DataArray_t");
  for (tree& coord : coords) {
    permute_node_value(value(coord),vertex_permutation);
  }
}


template<class I> auto
update_vertex_ids_in_connectivities(tree_range& elt_sections, const std::vector<I>& vertex_permutation) -> void {
  /* Precondition */ for([[maybe_unused]] tree& elt_section : elt_sections) { STD_E_ASSERT(label(elt_section)=="Elements_t"); }
  for(tree& elt_section : elt_sections) {
    re_number_vertex_ids_in_elements(elt_section,vertex_permutation);
  };
}


template<class I> auto
save_partition_point(tree& z, I nb_of_boundary_vertices) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  VertexBoundarySize_U<I>(z) = nb_of_boundary_vertices;
}


template<class I> auto
permute_boundary_vertices_at_beginning(tree& z, const std::vector<I>& boundary_vertex_ids) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  auto nb_of_vertices = VertexSize_U<I>(z);
  auto vertex_permutation = vertex_permutation_to_move_boundary_at_beginning(nb_of_vertices, boundary_vertex_ids);

  auto& grid_coords = get_child_by_name(z,"GridCoordinates");
  auto elt_sections = get_children_by_label(z,"Elements_t");
  permute_boundary_grid_coords_at_beginning(grid_coords,vertex_permutation);
  update_vertex_ids_in_connectivities(elt_sections,vertex_permutation);

  I vertex_partition_point = boundary_vertex_ids.size();
  save_partition_point(z,vertex_partition_point);
}


template<class I> auto
partition_coordinates(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (!is_boundary_partitioned_zone<I>(z)) {
    auto elt_sections = get_children_by_label(z,"Elements_t");
    auto boundary_vertex_ids = get_ordered_boundary_vertex_ids<I>(elt_sections);
    permute_boundary_vertices_at_beginning(z,boundary_vertex_ids);
  }
}


template<class I> auto
partition_elements(tree& z, cgns::donated_point_lists& plds) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  auto elt_sections = get_children_by_label(z,"Elements_t");
  tree& ngons = element_section(z,cgns::NGON_n);
  if (is_boundary_partitioned_element_section<I>(ngons)) return;

  // TODO
  //auto elts_permutation_0 = sort_ngons_by_nb_vertices(ngons);
  auto elts_permutation_1 = permute_boundary_ngons_at_beginning<I>(ngons);

  //mark_polygon_groups(ngons);

  //auto perm_new_to_old = std_e::compose_permutations(elts_permutation_1,elts_permutation_0);
  auto perm_new_to_old = elts_permutation_1;
  auto perm_old_to_new = std_e::inverse_permutation(perm_new_to_old);
  I offset = ElementRange<I>(ngons)[0];
  std_e::offset_permutation elts_perm(offset,perm_old_to_new);
  renumber_point_lists(z,elts_perm,"FaceCenter");
  renumber_point_lists_donated(plds,elts_perm,"FaceCenter");
}


template<class I> auto
_partition_zone_with_boundary_first(tree& z, cgns::donated_point_lists& plds) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (is_unstructured_zone(z)) {
    partition_coordinates<I>(z);
    partition_elements<I>(z,plds);
  }
}

auto
partition_zone_with_boundary_first(tree& z, cgns::donated_point_lists& plds) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (value(z).data_type()=="I4") return _partition_zone_with_boundary_first<I4>(z,plds);
  //if (value(z).data_type()=="I8") return _partition_zone_with_boundary_first<I8>(z,plds);
  if (value(z).data_type()=="I8") throw;
  throw cgns::cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}

auto
put_boundary_first(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(label(b)=="CGNSBase_t");
  apply_base_renumbering(b,partition_zone_with_boundary_first,comm); // TODO use Maia/CGNS standard for GC_t
}


} // maia
