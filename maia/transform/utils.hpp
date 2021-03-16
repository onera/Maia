#pragma once

#include <algorithm>
#include "cpp_cgns/sids/sids.hpp"
#include "pdm.h"
#include "mpi.h"
#include "maia/utils/parallel/distribution.hpp"

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

// TODO facto with partial_to_full_distribution
template<class Range> auto
distribution_from_partial(const Range& partial_distri, MPI_Comm comm) -> distribution_vector<PDM_g_num_t> {
  PDM_g_num_t dn_elt = partial_distri[1] - partial_distri[0];
  auto full_distri = distribution_from_dsizes(dn_elt, comm);
  STD_E_ASSERT(full_distri.back()==partial_distri[2]);
  return full_distri;
}

} // maia
