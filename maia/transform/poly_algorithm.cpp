#if __cplusplus > 201703L
#include "maia/transform/poly_algorithm.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include <algorithm>
#include <vector>
#include "cpp_cgns/sids/utils.hpp"
#include "maia/sids/element_sections.hpp"
#include "maia/transform/renumber/shift.hpp"
#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "std_e/data_structure/block_range/ivblock_range.hpp"

using namespace cgns;

namespace maia {

template<class I> auto
_ngon_new_to_old(tree& z) -> void {
  // 1. transform ngon/nface indexed->interleaved
  auto elt_sections = element_sections(z);
  for (tree& elt : elt_sections) {
    if (element_type(elt)==NGON_n || element_type(elt)==NFACE_n) {
      auto eso = ElementStartOffset<I>(elt);
      auto new_connectivity = ElementConnectivity<I>(elt);
      auto new_poly_range = std_e::view_as_vblock_range(new_connectivity,eso);

      I old_connectivity_sz = eso.size()-1 + eso.back();
      std::vector<I> old_connectivity(old_connectivity_sz);
      auto old_poly_range = std_e::view_as_ivblock_range(old_connectivity);

      std::ranges::copy(new_poly_range,old_poly_range.begin());

      rm_child_by_name(elt,"ElementConnectivity");
      rm_child_by_name(elt,"ElementStartOffset");

      emplace_child(elt,new_DataArray("ElementConnectivity",std::move(old_connectivity)));
    }
  }

  // 2. shift parent element to start at 1
  //    NOTE: this is done because Cassiopee expects ParentElements to start at 1
  auto face_sections = surface_element_sections(z);
  I n_faces = length(elements_interval(face_sections));
  shift_parent_elements(face_sections,-n_faces);
}

auto
ngon_new_to_old(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (value(z).data_type()=="I4") return _ngon_new_to_old<I4>(z);
  if (value(z).data_type()=="I8") return _ngon_new_to_old<I8>(z);
}


template<class I> auto
_ngon_old_to_new(tree& z) -> void {
  // 1. transform ngon/nface indexed->interleaved
  auto elt_sections = element_sections(z);
  for (tree& elt : elt_sections) {
    if (element_type(elt)==NGON_n || element_type(elt)==NFACE_n) {
      auto old_connectivity = ElementConnectivity<I>(elt);
      auto old_poly_range = std_e::view_as_ivblock_range(old_connectivity);

      auto elt_range = element_range(elt);
      auto n_elt = length(elt_range);
      std::vector<I> eso(n_elt+1);
      std::vector<I> new_connectivity(old_connectivity.size()-n_elt);
      auto new_poly_range = std_e::view_as_vblock_range(new_connectivity,eso);

      std::ranges::copy(old_poly_range,new_poly_range.begin());

      rm_child_by_name(elt,"ElementConnectivity");

      emplace_child(elt,new_DataArray("ElementStartOffset",std::move(eso)));
      emplace_child(elt,new_DataArray("ElementConnectivity",std::move(new_connectivity)));
    }
  }

  // 2. shift parent element to match the NFace ElementRange
  //    NOTE: this is done because in Cassiopee ParentElements start at 1
  auto face_sections = surface_element_sections(z);
  I first_cell_id = elements_interval(face_sections).last()+1;
  shift_parent_elements(face_sections,-1+first_cell_id); // -1 because old ParentElements starts at 1
}

auto
ngon_old_to_new(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (value(z).data_type()=="I4") return _ngon_old_to_new<I4>(z);
  if (value(z).data_type()=="I8") return _ngon_old_to_new<I8>(z);
}

} // maia
#endif // C++>17
