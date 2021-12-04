#include "maia/transform/__old/partition_with_boundary_first/partition_with_boundary_first.hpp"

#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices.hpp"
#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices_at_beginning.hpp"
#include "maia/transform/__old/partition_with_boundary_first/boundary_ngons_at_beginning.hpp"

#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "maia/transform/__old/base_renumbering.hpp"
#include "maia/transform/__old/renumber_point_lists.hpp"


namespace cgns {


auto
partition_with_boundary_first(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(label(b)=="CGNSBase_t");
  apply_base_renumbering(b,partition_zone_with_boundary_first,comm);
}


auto
partition_zone_with_boundary_first(tree& z, donated_point_lists& plds) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (is_unstructured_zone(z)) {
    partition_coordinates(z);
    partition_elements(z,plds);
  }
}


auto
partition_coordinates(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (!is_boundary_partitionned_zone<I4>(z)) {
    auto elt_pools = get_children_by_label(z,"Elements_t");
    auto boundary_vertex_ids = get_ordered_boundary_vertex_ids(elt_pools);
    permute_boundary_vertices_at_beginning(z,boundary_vertex_ids);
  }
}


auto
permute_boundary_vertices_at_beginning(tree& z, const std::vector<I4>& boundary_vertex_ids) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  auto nb_of_vertices = VertexSize_U<I4>(z);
  auto vertex_permutation = vertex_permutation_to_move_boundary_at_beginning(nb_of_vertices, boundary_vertex_ids);

  auto& grid_coords = get_child_by_name(z,"GridCoordinates");
  auto elt_pools = get_children_by_label(z,"Elements_t");
  permute_boundary_grid_coords_at_beginning(grid_coords,vertex_permutation);
  update_vertex_ids_in_connectivities(elt_pools,vertex_permutation);

  I4 vertex_partition_point = boundary_vertex_ids.size();
  save_partition_point(z,vertex_partition_point);
}


auto
permute_boundary_grid_coords_at_beginning(tree& grid_coords, const std::vector<I4>& vertex_permutation) -> void {
  STD_E_ASSERT(label(grid_coords)=="GridCoordinates_t");
  auto coords = get_children_by_label(grid_coords,"DataArray_t");
  for (tree& coord : coords) {
    permute_boundary_vertices(value(coord),vertex_permutation);
  }
}

auto
permute_boundary_vertices(node_value& coord, const std::vector<I4>& perm) -> void {
  auto coord_span = view_as_span<R8>(coord);
  std_e::permute(coord_span.begin(), perm);
};


auto
update_vertex_ids_in_connectivities(tree_range& elt_pools, const std::vector<I4>& vertex_permutation) -> void {
  /* Precondition */ for([[maybe_unused]] tree& elt_pool : elt_pools) { STD_E_ASSERT(label(elt_pool)=="Elements_t"); }
  for(tree& elt_pool : elt_pools) {
    re_number_vertex_ids_in_elements(elt_pool,vertex_permutation);
  };
}


auto
save_partition_point(tree& z, I4 nb_of_boundary_vertices) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  VertexBoundarySize_U<I4>(z) = nb_of_boundary_vertices;
}


auto
partition_elements(tree& z, donated_point_lists& plds) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  auto elt_pools = get_children_by_label(z,"Elements_t");
  tree& ngons = element_section(z,NGON_n);
  if (is_boundary_partitionned_element_pool<I4>(ngons)) return;

  auto elts_permutation_0 = sort_ngons_by_nb_vertices(ngons);
  auto elts_permutation_1 = permute_boundary_ngons_at_beginning(ngons);

  mark_polygon_groups(ngons);

  auto perm_new_to_old = std_e::compose_permutations(elts_permutation_1,elts_permutation_0);
  auto perm_old_to_new = std_e::inverse_permutation(perm_new_to_old);
  I4 offset = ElementRange<I4>(ngons)[0];
  std_e::offset_permutation elts_perm(offset,perm_old_to_new);
  renumber_point_lists(z,elts_perm,"FaceCenter");
  renumber_point_lists_donated(plds,elts_perm,"FaceCenter");
}


} // cgns
