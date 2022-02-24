#include "maia/sids/maia_cgns.hpp"

#include "maia/sids/element_sections.hpp"


using namespace cgns;


namespace maia {


auto
is_maia_cgns_zone(const cgns::tree& z) -> bool {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (ZoneType(z)=="Unstructured") {
    auto face_sections = surface_element_sections(z);
    auto cell_sections = volume_element_sections(z);
    auto last_2d_elt_id = elements_interval(face_sections).last();
    auto first_3d_elt_id = elements_interval(cell_sections).first();
    if (last_2d_elt_id+1!=first_3d_elt_id) return false;
  }
  return true;
}


} // maia
