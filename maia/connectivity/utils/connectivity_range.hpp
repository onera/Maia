#pragma once


#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"


namespace maia {


template<class I, class Tree>
  requires (std::is_same_v< std::remove_const_t<Tree> , cgns::tree >)
    auto
make_connectivity_range(Tree& elt_section) {
  auto cs      = cgns::ElementConnectivity<I>(elt_section);
  auto offsets = cgns::ElementStartOffset <I>(elt_section);
  return std_e::view_as_vblock_range2(cs,offsets);
}
template<class I, class Tree>
  requires (std::is_same_v< std::remove_const_t<Tree> , cgns::tree >)
    auto
make_connectivity_subrange(Tree& elt_section, I start, I finish) {
  auto cs      = cgns::ElementConnectivity<I>(elt_section);
  auto offsets = cgns::ElementStartOffset <I>(elt_section);

  auto sub_offsets = std_e::make_span(offsets.data()+start,finish-start+1);
  return std_e::view_as_vblock_range2(cs,sub_offsets);
}


} // maia
