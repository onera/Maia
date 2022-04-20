#pragma once


#include "pdm.h"
#include <vector>
#include "std_e/algorithm/distribution.hpp"
#include "std_e/interval/interval_sequence.hpp"
#include "std_e/parallel/mpi.hpp"

/*
concept Distribution : std_e::Interval_sequence
  value_type is Integer
  the interval starts at 0
*/
// TODO maybe a Distribution is more than that: hold the communicator?

template<class Integer> using distribution_vector = std_e::interval_vector<Integer>;
template<class Integer> using distribution_span = std_e::interval_span<Integer>;


// Distribution vocabulary around interval_sequence functions {
template<class Integer, class Distribution> auto
search_rank(Integer i, const Distribution& dist) {
  return std_e::interval_index(i,dist);
}

template<class Random_access_range, class I = typename Random_access_range::value_type> auto
distribution_from_sizes(const Random_access_range& r) -> distribution_vector<I> {
  return std_e::indices_from_strides(r);
}
// Distribution vocabulary around interval_sequence functions }

template<class Integer> auto
distribution_from_dsizes(Integer dn, MPI_Comm comm) -> distribution_vector<Integer> {
  int n_rank = std_e::n_rank(comm);
  std::vector<Integer> dn_elts(n_rank);
  MPI_Allgather((void*) &dn           , 1, std_e::to_mpi_type<Integer>,
                (void*)  dn_elts.data(), 1, std_e::to_mpi_type<Integer>,
                comm);
  return std_e::indices_from_strides(dn_elts);
}

template<class I>
distribution_vector<I> uniform_distribution(int size, I nb_elts) {
  distribution_vector<I> distrib(size);
  std_e::uniform_distribution(begin(distrib),end(distrib),nb_elts);
  return distrib;
}

template<typename Container, typename T = typename Container::value_type>
void compute_multi_range_from_count(Container& x, T start){
  T tmp1 = x[0];
  x[0] = start;
  for(int i = 0; i < static_cast<int>(x.size())-1; ++i){
    T tmp2 = x[i+1];
    x[i+1] = x[i] + tmp1;
    tmp1 = tmp2;
  }
}


//Content of maia/utils/parallel/utils.hpp
namespace maia {


template<class Range> auto
partial_to_full_distribution(const Range& partial_distrib, MPI_Comm comm) {
  STD_E_ASSERT(partial_distrib.size()==3);
  using I = typename Range::value_type;

  distribution_vector<I> full_distrib(std_e::n_rank(comm));
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
