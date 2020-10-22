#pragma once


#include "cpp_cgns/node_manip.hpp"


namespace cgns {


auto ngon_boundary_vertices(std_e::span<const I4> connectivities, md_array_view<const I4,2> parent_elts) -> std::vector<I4>;

auto get_elements_boundary_vertices(const tree& elts) -> std::vector<I4>;
auto append_boundary_coordinates_indices(const tree& elts, std::vector<I4>& boundary_vertex_indices) -> void;
auto get_ordered_boundary_vertex_ids(const tree_range& elements_range) -> std::vector<I4>;


} // cgns
