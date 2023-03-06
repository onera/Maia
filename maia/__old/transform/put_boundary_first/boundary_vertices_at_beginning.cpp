#if __cplusplus > 201703L
#include "maia/__old/transform/put_boundary_first/boundary_vertices_at_beginning.hpp"

#include "maia/utils/logging/log.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"


using cgns::tree;
using cgns::I4;
using cgns::I8;


namespace maia {


template<class I> auto
vertex_permutation_to_move_boundary_at_beginning(I n_vtx, const std::vector<I>& boundary_vertex_ids) -> std::vector<I> {
  std::vector<bool> vertices_are_on_boundary(n_vtx,false);
  for (auto boundary_vertex_id : boundary_vertex_ids) {
    I boundary_vertex_index = boundary_vertex_id - 1; // C++ is 0-indexed, CGNS ids are 1-indexed
    vertices_are_on_boundary[boundary_vertex_index] = true;
  }

  auto vertex_permutation = std_e::iota_vector(n_vtx); // init with no permutation
  std::ranges::partition(vertex_permutation,[&](auto i){ return vertices_are_on_boundary[i]; });
  return vertex_permutation;
}


template<class I> auto
update_ids_for_elt_type(tree& elt_section, const std::vector<I>& vertex_permutation) -> void {
  auto elt_vtx = cgns::ElementConnectivity<I>(elt_section);

  auto perm_old_to_new = std_e::inverse_permutation(vertex_permutation);
  I offset = 1; // CGNS ids begin at 1
  std_e::offset_permutation perm(offset,perm_old_to_new);
  std_e::apply(perm,elt_vtx);
}

template<class I> auto
re_number_vertex_ids_in_elements(tree& elt_section, const std::vector<I>& vertex_permutation) -> void {
  // Preconditions
  //   - vertex_permutation is an index permutation (i.e. sort(permutation) == std_e::iota(permutation.size()))
  //   - any vertex "v" of "elt_section" is referenced in "vertex_permutation",
  //     i.e. vertex_permutation[v-1] is valid ("-1" because of 1-indexing)
  auto _ = maia_time_log("re_number_vertex_ids_in_elements");

  update_ids_for_elt_type(elt_section,vertex_permutation);
}


// explicit instanciations (do not pollute the header for only 2 instanciations)
template auto vertex_permutation_to_move_boundary_at_beginning(I4 n_vtx, const std::vector<I4>& boundary_vertex_ids) -> std::vector<I4>;
template auto vertex_permutation_to_move_boundary_at_beginning(I8 n_vtx, const std::vector<I8>& boundary_vertex_ids) -> std::vector<I8>;
template auto re_number_vertex_ids_in_elements(tree& elt_section, const std::vector<I4>& vertex_permutation) -> void;
template auto re_number_vertex_ids_in_elements(tree& elt_section, const std::vector<I8>& vertex_permutation) -> void;

} // maia
#endif // C++>17
