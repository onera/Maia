#pragma once


#include "cpp_cgns/cgns.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "maia/transform/__old/donated_point_lists.hpp"
#include "cpp_cgns/sids/creation.hpp"


namespace cgns {


auto renumber_point_lists(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void;
auto renumber_point_lists2(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void; // TODO DEL
auto renumber_point_lists_donated(donated_point_lists& plds, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void;

auto
rm_invalid_ids_in_point_lists(tree& z, const std::string& grid_location, factory F) -> void;
auto
rm_invalid_ids_in_point_lists_with_donors(tree& z, const std::string& grid_location, factory F) -> void;
auto
rm_grid_connectivities(tree& z, const std::string& grid_location, factory F) -> void;

} // cgns
