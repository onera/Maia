#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "maia/algo/dist/std_elements_to_ngons/interior_faces_and_parents/struct/faces_and_parents_by_section.hpp"


namespace maia {


template<class I> auto
generate_element_faces_and_parents(const cgns::tree_range& elt_sections) -> faces_and_parents_by_section<I>;


} // maia
