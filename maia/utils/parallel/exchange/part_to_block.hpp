#pragma once


#include "std_e/parallel/serialize.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/utils/parallel/exchange/global_numbering.hpp"
#include "std_e/algorithm/unique_compress.hpp"
#include "pdm_part_to_block.h"
#include <algorithm>
#include <mpi.h>
#include <cassert>


namespace pdm {


template<PDM_part_to_block_post_t post_treatment_kind>
class parts_to_block_protocol {
  public:
    parts_to_block_protocol(MPI_Comm mpi_comm, distribution_vector<PDM_g_num_t> dist, global_numberings lgs)
      : distribution(move(dist))
      , LN_to_GNs(move(lgs))
      , nb_elts_by_partition(LN_to_GNs.size())
      , g_num_by_partition(LN_to_GNs.size())
    {
      // for (auto& LN_to_GN : LN_to_GNs) {
      //   std_e::offset(LN_to_GN,1); // TODO why does Paradigma uses 1-indexed LN_to_GN but 0-indexed distribution ???
      // }
      std::transform(begin(LN_to_GNs),end(LN_to_GNs),begin(nb_elts_by_partition),[](auto& LN_to_GN){ return LN_to_GN.size(); });
      std::transform(begin(LN_to_GNs),end(LN_to_GNs),begin(g_num_by_partition),[](auto& LN_to_GN){ return LN_to_GN.data(); });

      PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&mpi_comm);

      int n_part = LN_to_GNs.size();

      ptb = PDM_part_to_block_create_from_distrib( PDM_PART_TO_BLOCK_DISTRIB_ALL_PROC,
                                       post_treatment_kind,
                                       0.,
                                       g_num_by_partition.data(),
                                       distribution.data(),
                                       nb_elts_by_partition.data(),
                                       n_part,
                                       pdm_comm);
    }


    template<class Contiguous_range>
    /// requires Contiguous_range::value_type==std_e::serialized_array
    std_e::serialized_array exchange(Contiguous_range& arrays_by_partition) {
      int n_part = LN_to_GNs.size();
      std::vector<std::vector<int>> part_strides(n_part);
      for (int i=0; i<n_part; ++i) {
        part_strides[i] = interval_lengths(arrays_by_partition[i].offsets);
      }
      std::vector<int*> part_stride_pt(n_part);
      std::vector<std::byte*> part_data_pt(n_part);
      std::transform(begin(part_strides),end(part_strides),begin(part_stride_pt),[](auto& part_stride){ return part_stride.data(); });
      std::transform(begin(arrays_by_partition),end(arrays_by_partition),begin(part_data_pt),[](auto& arrays){ return arrays.data.data(); });

      int*  blk_strid;
      std::byte* blk_data;
      int nMaxSize = PDM_part_to_block_exch(         ptb,
                                                     sizeof(std::byte),
                                                     PDM_STRIDE_VAR_INTERLACED,
                                                     -1,
                                                     part_stride_pt.data(),
                                            (void**) part_data_pt.data(),
                                                     &blk_strid,
                                            (void**) &blk_data);
      assert(nMaxSize >= 0);

      // TODO rather than copying to std::vectors ONLY to replace manual calls to free by automatic RAII,
      // it would be better to either
      //  - change the PDM_part_to_block_exch interface to accept vector
      //  - change the PDM_part_to_block_exch interface behavior so that we allocate memory ourselves
      int block_size = PDM_part_to_block_n_elt_block_get(ptb);
      auto offsets = indices_from_strides(std_e::make_span(blk_strid,block_size));

      // TODO Bruno paradigm function to get that ? (paradigm knows it for malloc)
      int block_cat_size = offsets.back();
      std::vector<std::byte> cat_arrays_block(block_cat_size);
      std::copy_n(blk_data,block_cat_size,begin(cat_arrays_block));

      free(blk_strid);
      free(blk_data);

      return {offsets,cat_arrays_block};
    }


    template<class T, class F>
    std::vector<T> exchange2(const std::vector<std::vector<T>>& arrays_by_partition, F reduction) {
      int n_part = LN_to_GNs.size();
      std::vector<std::vector<int>> part_strides(n_part);
      for (int i=0; i<n_part; ++i) {
        part_strides[i] = std::vector<int>(arrays_by_partition[i].size(),1);
      }
      std::vector<int*> part_stride_pt(n_part);
      std::vector<const T*> part_data_pt(n_part);
      std::transform(begin(part_strides),end(part_strides),begin(part_stride_pt),[](auto& part_stride){ return part_stride.data(); });
      std::transform(begin(arrays_by_partition),end(arrays_by_partition),begin(part_data_pt),[](auto& arrays){ return arrays.data(); });

      int*  blk_strid;
      T* blk_data;
      int nMaxSize = PDM_part_to_block_exch(         ptb,
                                                     sizeof(T),
                                                     PDM_STRIDE_VAR_INTERLACED,
                                                     -1,
                                                     part_stride_pt.data(),
                                            (void**) part_data_pt.data(),
                                                     &blk_strid,
                                            (void**) &blk_data);
      assert(nMaxSize >= 0);

      // TODO rather than copying to std::vectors ONLY to replace manual calls to free by automatic RAII,
      // it would be better to either
      //  - change the PDM_part_to_block_exch interface to accept vector
      //  - change the PDM_part_to_block_exch interface behavior so that we allocate memory ourselves
      int block_size = PDM_part_to_block_n_elt_block_get(ptb);
      auto offsets = indices_from_strides(std_e::make_span(blk_strid,block_size));

      // TODO Bruno paradigm function to get that ? (paradigm knows it for malloc)
      int block_cat_size = offsets.back();
      std::vector<T> cat_arrays_block(block_cat_size);

      auto last = std_e::unique_compress_strides_copy(blk_data,blk_data+block_cat_size,begin(cat_arrays_block),reduction,blk_strid);
      cat_arrays_block.resize(last-begin(cat_arrays_block));

      free(blk_strid);
      free(blk_data);

      return cat_arrays_block;
    }

    ~parts_to_block_protocol() {
      PDM_part_to_block_free(ptb);
    }

    // Don't know how PDM_part_to_block_t reacts to address changes
    // So prevent these unsafe operations
    parts_to_block_protocol(parts_to_block_protocol&&) = delete;
    parts_to_block_protocol& operator=(parts_to_block_protocol&&) = delete;
    parts_to_block_protocol(const parts_to_block_protocol&) = delete;
    parts_to_block_protocol& operator=(const parts_to_block_protocol&) = delete;

  private:
    distribution_vector<PDM_g_num_t> distribution;
    global_numberings LN_to_GNs;

    std::vector<int> nb_elts_by_partition;
    std::vector<PDM_g_num_t*> g_num_by_partition;

    PDM_part_to_block_t* ptb;
};


template<PDM_part_to_block_post_t post_treatment_kind>
class part_to_block_protocol {
  public:
    part_to_block_protocol(MPI_Comm mpi_comm, distribution_vector<PDM_g_num_t> dist, global_numbering lg)
      : impl(mpi_comm,std::move(dist),{std::move(lg)})
    {}

    std_e::serialized_array exchange(std_e::serialized_array& arrays) {
      std::vector<std_e::serialized_array> arrays_by_partition = {arrays}; // TODO remove copy
      return impl.exchange(arrays_by_partition);
    }
    template<class T, class F>
    std::vector<T> exchange2(const std::vector<T>& arrays, F reduction) {
      std::vector<std::vector<T>> arrays_by_partition = {arrays}; // TODO remove copy
      return impl.exchange2(arrays_by_partition,reduction);
    }
  private:
    parts_to_block_protocol<post_treatment_kind> impl;
};


using parts_to_unique_block_protocol = parts_to_block_protocol<PDM_PART_TO_BLOCK_POST_CLEANUP>;
using parts_to_merged_block_protocol = parts_to_block_protocol<PDM_PART_TO_BLOCK_POST_MERGE>;

using part_to_unique_block_protocol = part_to_block_protocol<PDM_PART_TO_BLOCK_POST_CLEANUP>;
using part_to_merged_block_protocol = part_to_block_protocol<PDM_PART_TO_BLOCK_POST_MERGE>;


template<PDM_part_to_block_post_t post, class Contiguous_range, class T = typename Contiguous_range::value_type> auto
exchange(part_to_block_protocol<post>& ptb, const Contiguous_range& part_array) -> std::vector<T> {
  auto part_serial = std_e::serialize_array(part_array);
  auto dist_serial =  ptb.exchange(part_serial);
  return std_e::deserialize_array<T>(dist_serial);
}
template<PDM_part_to_block_post_t post, class T, class F> auto
exchange2(part_to_block_protocol<post>& ptb, const std::vector<T>& part_array, F reduction) -> std::vector<T> {
  return ptb.exchange2(part_array,reduction);
}
//template<class T> auto
//exchange(block_to_parts_protocol& btp, std::vector<T>& dist_array) -> std::vector<std::vector<T>> {
//  auto dist_serial = std_e::serialize_array(dist_array);
//  auto parts_serial =  btp.exchange(dist_serial);
//
//  int n_parts = parts_serial.size();
//  std::vector<std::vector<T>> res(n_parts);
//  for (int i=0; i<n_parts; ++i) {
//    res[i] = std_e::deserialize_array<T>(parts_serial[i]);
//  }
//  return res;
//}


} // pdm
