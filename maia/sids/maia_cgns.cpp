#include "maia/sids/maia_cgns.hpp"

#include "maia/sids/element_sections.hpp"


using cgns::tree;


namespace maia {


auto
zone_sections_are_ordered_by_increasing_dimensions(const tree& z) {
  if (ZoneType(z)=="Unstructured") {
    auto face_sections = surface_element_sections(z);
    auto cell_sections = volume_element_sections(z);
    auto last_2d_elt_id = elements_interval(face_sections).last();
    auto first_3d_elt_id = elements_interval(cell_sections).first();
    if (last_2d_elt_id+1!=first_3d_elt_id) return false;
  }
  return true;
}


auto
is_maia_compliant_zone(const tree& z) -> bool {
  STD_E_ASSERT(label(z)=="Zone_t");
  return zone_sections_are_ordered_by_increasing_dimensions(z);
  // TODO + at most one section per element type
  //      + GC 1to1 with name of opposite GC
}


} // maia
