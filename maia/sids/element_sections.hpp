#pragma once

#include <algorithm>
#include "cpp_cgns/sids/sids.hpp"

namespace maia {


template<class Tree> auto
element_sections_ordered_by_range(Tree& z) {
  auto elt_sections = get_children_by_label(z,"Elements_t");
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  return elt_sections;
}
template<class Tree> auto
element_sections_ordered_by_range_by_type(Tree& z) {
  auto elt_sections = get_children_by_label(z,"Elements_t");
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_elt_type);
  return elt_sections;
}


} // maia
