#pragma once


#include "cpp_cgns/cgns.hpp"
#include "mpi.h"


#include "std_e/data_structure/jagged_range.hpp"
#include "std_e/future/contract.hpp"
#include "std_e/algorithm/partition_sort.hpp"
#include "std_e/parallel/all_to_all.hpp"


namespace maia {

auto add_fsdm_distribution(cgns::tree& b, MPI_Comm comm) -> void;

auto distribute_bc_ids_to_match_face_dist(cgns::tree& b, MPI_Comm comm) -> void;
auto distribute_vol_fields_to_match_global_element_range(cgns::tree& b, MPI_Comm comm) -> void;
auto distribute_fields_to_match_global_element_range(cgns::tree& b, MPI_Comm comm) -> void;


// distribute_bc_ids_to_match_face_dist impl {
// Note: see this as a temporary until a multi_array from a "rack" is implemented
template<class Range_of_ranges> auto
_transposed(const Range_of_ranges& x) {
  // Precond:
  //   all ranges have the same size
  //   there is at least one range
  using range_type = typename Range_of_ranges::value_type;
  using T = typename range_type::value_type;
  int m = x.size();
  int n = x[0].size();
  std::vector<std::vector<T>> xt(n);
  for (int j=0; j<n; ++j) {
    xt[j] = std::vector<T>(m);
    for (int i=0; i<m; ++i) {
      xt[j][i] = x[i][j];
    }
  }
  return xt;
}


template<class Dist_range, class Interval_sequence, class Range, class Range_of_ranges> auto
repartition_by_distributions(
  const Dist_range& elt_dists,
  const Interval_sequence& elt_intervals,
  Range& point_list,
  Range_of_ranges& values
)
{
  using I = typename Range::value_type;
  std::vector<I> perm_indices(point_list.size());
  std::iota(begin(perm_indices),end(perm_indices),0);

  // Note: could be done in place, but needs more infrastructure
  auto multi_dist = _transposed(elt_dists);
  int n_dist = multi_dist.size()-1; // TODO length
  auto multi_dist_intervals = std_e::make_span(multi_dist.data()+1,n_dist);

  auto comp = [&elt_intervals,&point_list](I perm_indices, const auto& dist_interval){
    I id = point_list[perm_indices];
    int idx = std_e::interval_index(id,elt_intervals);
    I base_id = id - elt_intervals[idx];
    return base_id < dist_interval[idx];
  };

  auto partition_indices = std_e::partition_sort_indices(perm_indices,multi_dist_intervals,comp);

  std_e::permute(begin(point_list),perm_indices);
  for (auto& value : values) {
    std_e::permute(begin(value),perm_indices);
  }

  return partition_indices;
}


template<class Dist_range, class Interval_sequence, class Range, class Range_of_ranges> auto
redistribute_to_match_face_dist(
  const Dist_range& elt_dists,
  const Interval_sequence& elt_intervals,
  Range& point_list,
  Range_of_ranges& values,
  MPI_Comm comm
)
{
  using I = typename Range::value_type;

  auto partition_indices = repartition_by_distributions(elt_dists,elt_intervals,point_list,values);

  std_e::jagged_span<I,2> pl(std_e::make_span(point_list),std_e::make_span(partition_indices));
  std::vector<I> pl_new = std_e::all_to_all_v(pl,comm).retrieve_values();
  I pl_size = pl_new.size();

  int n_value = values.size();
  using value_range = typename Range_of_ranges::value_type;
  using T = typename value_range::value_type;
  std::vector<std::vector<T>> values_new(n_value);
  for (int i=0; i<n_value; ++i) {
    std_e::jagged_span<T,2> val(std_e::make_span(values[i]),std_e::make_span(partition_indices));
    values_new[i] = std_e::all_to_all_v(val,comm).retrieve_values();
  }

  std::vector<I> partial_dist(3);
  partial_dist[0] = std_e::ex_scan(pl_size,MPI_SUM,0,comm);
  partial_dist[1] = partial_dist[0] + pl_size;
  partial_dist[2] = std_e::all_reduce(pl_size,MPI_SUM,comm);
  return std::make_tuple(std::move(partial_dist),std::move(pl_new),std::move(values_new));
}
// distribute_bc_ids_to_match_face_dist impl }

} // maia
