#pragma once


#include "cpp_cgns/cgns.hpp"


namespace maia {


  // TODO test
template<class I> auto
permute_parent_elements(cgns::md_array_view<I,2>& parent_elts, const std::vector<I>& permutation) -> void {
  std_e::permute(column(parent_elts,0).begin(),permutation);
  std_e::permute(column(parent_elts,1).begin(),permutation);
}
template<class I> auto
inv_permute_parent_elements(cgns::md_array_view<I,2>& parent_elts, const std::vector<I>& permutation, I offset) -> void {
  auto inv_p = std_e::inverse_permutation(permutation);
  std_e::offset_permutation op(offset,inv_p);
  auto not_bnd_parent = [](I pe){ return pe != 0; };
  std_e::apply_if(op,parent_elts,not_bnd_parent);
}
template<class I> auto // TODO factor with above
inv_permute_parent_elements_sub(cgns::md_array_view<I,2>& parent_elts, const std::vector<I>& permutation, I offset, I inf, I sup) -> void {
  auto inv_p = std_e::inverse_permutation(permutation);
  std_e::offset_permutation op(offset,inv_p);
  auto in_inter = [inf,sup](I pe){ return pe!=0 && inf <= pe && pe < sup; };
  std_e::apply_if(op,parent_elts,in_inter);
}
template<class I> auto
inv_permute_connectivity(auto& cs, const std::vector<I>& permutation, I offset) -> void {
  auto inv_p = std_e::inverse_permutation(permutation);
  std_e::offset_permutation op(offset,inv_p);
  std_e::apply(op,cs);
}
template<class I> auto
inv_permute_connectivity_sub(auto& cs, const std::vector<I>& permutation, I offset) -> void { // TODO rename _start??
  auto inv_p = std_e::inverse_permutation(permutation);
  std_e::offset_permutation op(offset,inv_p);
  I last_id = offset+permutation.size();
  auto affected_by_permutation = [last_id](I i){ return i < last_id; };
  std_e::apply_if(op,cs,affected_by_permutation);
}


} // maia
