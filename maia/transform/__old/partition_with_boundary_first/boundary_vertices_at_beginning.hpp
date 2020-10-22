#pragma once

#include <vector>
#include "cpp_cgns/cgns.hpp"

namespace cgns {

template<class I> auto
vertex_permutation_to_move_boundary_at_beginning(I nb_of_vertices, const std::vector<I>& boundary_vertex_ids) -> std::vector<I>;

template<class I> auto
re_number_vertex_ids_in_elements(tree& elt_pool, const std::vector<I>& vertex_permutation) -> void;

} // cgns
