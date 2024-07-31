#pragma once


#include "cpp_cgns/sids.hpp"


namespace maia {


template<class Tree_range, class I> auto
shift_element_ranges(Tree_range& elt_sections, I offset) -> void {
  for (cgns::tree& elt_section : elt_sections) {
    auto elt_interval = ElementRange<I>(elt_section);
    elt_interval[0] += offset;
    elt_interval[1] += offset;
  }
}
template<class Tree_range, class I> auto
shift_parent_elements(Tree_range& face_sections, I offset) -> void {
  for (cgns::tree& face_section : face_sections) {
    if (has_child_of_name(face_section,"ParentElements")) {
      auto pe = get_node_value_by_matching<I,2>(face_section,"ParentElements");
      for (I& id : pe) {
        if (id != 0) {
          id += offset;
        }
      }
    }
  }
}
template<class I> auto
shift_cell_ids(cgns::tree& z, I offset) -> void {
  auto cell_sections = volume_element_sections(z);
  shift_element_ranges(cell_sections,offset);

  auto face_sections = surface_element_sections(z);
  shift_parent_elements(face_sections,offset);

  // TODO also shift all PointList located at CellCenter
}


} // maia
