#include "std_e/parallel/mpi.hpp"
#include <vector>
#include "cpp_cgns/sids/sids.hpp"
#include "pdm.h"
#include "mpi.h"
#include "maia/utils/parallel/distribution.hpp"


namespace maia {


template<class Range> auto
partial_to_full_distribution(const Range& partial_distrib, MPI_Comm comm) {
  STD_E_ASSERT(partial_distrib.size()==3);
  using I = typename Range::value_type;

  distribution_vector<I> full_distrib(std_e::nb_ranks(comm));
  full_distrib[0] = 0;
  std_e::all_gather(partial_distrib[1], full_distrib.data()+1, comm);

  STD_E_ASSERT(full_distrib[std_e::rank(comm)  ] == partial_distrib[0]);
  STD_E_ASSERT(full_distrib[std_e::rank(comm)+1] == partial_distrib[1]);
  STD_E_ASSERT(full_distrib.back()               == partial_distrib[2]);
  return full_distrib;
}
template<class Range> auto
full_to_partial_distribution(const Range& full_distrib, MPI_Comm comm) {
  using I = typename Range::value_type;

  std::vector<I> partial_distrib(3);
  partial_distrib[0] = full_distrib[std_e::rank(comm)  ];
  partial_distrib[1] = full_distrib[std_e::rank(comm)+1];
  partial_distrib[2] = full_distrib.back()              ;
  return partial_distrib;
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
