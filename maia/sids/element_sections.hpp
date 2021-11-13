#pragma once

#include <algorithm>
#include "cpp_cgns/sids/sids.hpp"

namespace maia {

template<class Tree> auto
element_sections(Tree& z) {
  return get_children_by_label(z,"Elements_t");
}

auto is_section_of_dimension(const cgns::tree& n, int dim) -> bool;

template<class Tree> auto
element_sections_of_dim(Tree& z, int dim) {
  auto is_section_of_dim = [dim](const cgns::tree& n) -> bool { return is_section_of_dimension(n,dim); };
  return get_children_by_predicate(z,is_section_of_dim);
}
template<class Tree> auto
surface_element_sections(Tree& z) {
  return element_sections_of_dim(z,2);
}
template<class Tree> auto
volume_element_sections(Tree& z) {
  return element_sections_of_dim(z,3);
}

template<class Tree> auto
element_sections_ordered_by_range(Tree& z) {
  auto elt_sections = element_sections(z);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  return elt_sections;
}
template<class Tree> auto
element_sections_ordered_by_range_by_type(Tree& z) {
  auto elt_sections = element_sections(z);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_elt_type);
  return elt_sections;
}

auto max_element_id(const cgns::tree& z) -> cgns::I8;

auto surface_elements_interval(const cgns::tree& z) -> cgns::interval<cgns::I8>;
auto volume_elements_interval(const cgns::tree& z) -> cgns::interval<cgns::I8>;


} // maia
