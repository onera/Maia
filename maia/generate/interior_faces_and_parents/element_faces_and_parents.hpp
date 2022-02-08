#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "maia/generate/interior_faces_and_parents/struct.hpp"


namespace maia {


template<class I, class Tree_range> auto
generate_element_faces_and_parents(const Tree_range& elt_sections) -> faces_and_parents_by_section<I>;


} // maia
