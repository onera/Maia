#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "maia/generate/interior_faces_and_parents/faces_and_parents_by_section.hpp"


namespace maia {


template<class I> auto
generate_element_faces_and_parents(const cgns::tree_range& elt_sections) -> faces_and_parents_by_section<I>;


} // maia
