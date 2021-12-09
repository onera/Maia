#include "maia/sids/maia_cgns.hpp"

#include "maia/sids/element_sections.hpp"


using namespace cgns;


namespace maia {


auto
is_maia_cgns_zone(const cgns::tree& z) -> bool {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (ZoneType(z)=="Unstructured") {
    auto last_2d_elt_id = surface_elements_interval(z).last();
    auto first_3d_elt_id = volume_elements_interval(z).first();
    if (last_2d_elt_id+1!=first_3d_elt_id) return false;
  }
  return true;
}


} // maia
