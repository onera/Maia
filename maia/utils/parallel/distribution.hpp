#pragma once


#include "std_e/algorithm/distribution.hpp"
#include "std_e/interval/interval_sequence.hpp"
#include "std_e/parallel/mpi.hpp"

/*
concept Distribution : std_e::Knot_sequence
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
