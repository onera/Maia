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


} // maia
