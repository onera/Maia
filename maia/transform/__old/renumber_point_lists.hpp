#pragma once


#include "cpp_cgns/cgns.hpp"
#include "std_e/algorithm/id_permutations.hpp"
#include "maia/transform/__old/donated_point_lists.hpp"


namespace maia {

template<class I> auto
renumber_point_lists(const std::vector<std_e::span<I>>& pls, const std_e::offset_permutation<I>& permutation) -> void;

template<class I>
auto renumber_point_lists(cgns::tree& z, const std_e::offset_permutation<I>& permutation, const std::string& grid_location) -> void;

auto renumber_point_lists2(cgns::tree& z, const std_e::offset_permutation<cgns::I4>& permutation, const std::string& grid_location) -> void; // TODO DEL
auto renumber_point_lists_donated(donated_point_lists& plds, const std_e::offset_permutation<cgns::I4>& permutation, const std::string& grid_location) -> void;

auto
rm_invalid_ids_in_point_lists(cgns::tree& z, const std::string& grid_location) -> void;
auto
rm_invalid_ids_in_point_lists_with_donors(cgns::tree& z, const std::string& grid_location) -> void;
auto
rm_grid_connectivities(cgns::tree& z, const std::string& grid_location) -> void;

} // maia
