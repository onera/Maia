#if __cplusplus > 201703L
#include "maia/algo/common/poly_algorithm.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include <algorithm>
#include <vector>
#include "cpp_cgns/sids/utils.hpp"
#include "maia/pytree/maia/element_sections.hpp"
#include "maia/algo/common/shift.hpp"
#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "std_e/data_structure/block_range/ivblock_range.hpp"

using namespace cgns;

namespace maia {

template<class I> auto
_indexed_to_interleaved_connectivity(tree& elt) -> void {
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

auto
indexed_to_interleaved_connectivity(tree& elt) -> void {
  STD_E_ASSERT(label(elt)=="Elements_t");
  if (value(elt).data_type()=="I4") return _indexed_to_interleaved_connectivity<I4>(elt);
  if (value(elt).data_type()=="I8") return _indexed_to_interleaved_connectivity<I8>(elt);
}


template<class I> auto
_interleaved_to_indexed_connectivity(tree& elt) -> void {
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

auto
interleaved_to_indexed_connectivity(tree& elt) -> void {
  STD_E_ASSERT(label(elt)=="Elements_t");
  if (value(elt).data_type()=="I4") return _interleaved_to_indexed_connectivity<I4>(elt);
  if (value(elt).data_type()=="I8") return _interleaved_to_indexed_connectivity<I8>(elt);
}

} // maia
#endif // C++>17
