#pragma once

#include "maia/utils/parallel/exchange/part_to_block.hpp"
#include "maia/utils/parallel/exchange/block_to_part.hpp"


template<class Contiguous_range, class Integer_contiguous_range0, class Integer_contiguous_range1, class T = typename Contiguous_range::value_type> auto
spread_then_collect(
  MPI_Comm comm, const distribution_vector<PDM_g_num_t>& distribution,
  Integer_contiguous_range0&& LN_to_GN_spread, const Contiguous_range& data_to_spread,
  Integer_contiguous_range1&& LN_to_GN_collect
) -> std::vector<T>
{
  pdm::part_to_unique_block_protocol ptb_protocol(comm,distribution,FWD(LN_to_GN_spread));
  std::vector<T> dist_data = pdm::exchange(ptb_protocol,data_to_spread);

  pdm::block_to_part_protocol btp_protocol(comm,distribution,FWD(LN_to_GN_collect));
  return pdm::exchange(btp_protocol,dist_data);
}


//template<class T, class F> auto
//spread_then_collect_multiple(
//  MPI_Comm comm, const distribution_vector<int>& distribution,
//  std::vector<int> LN_to_GN_spread, std::vector<int> LN_to_GN_collect,
//  const std::vector<T>& part_data,
//  F reduction
//) -> std::vector<T>
//{
//  pdm::part_to_merged_block_protocol ptb_protocol(comm,distribution,std::move(LN_to_GN_spread));
//  std::vector<T> dist_data = pdm::exchange2(ptb_protocol,part_data,reduction);
//
//  pdm::block_to_part_protocol btp_protocol(comm,distribution,std::move(LN_to_GN_collect));
//  return pdm::exchange(btp_protocol,dist_data);
//}
