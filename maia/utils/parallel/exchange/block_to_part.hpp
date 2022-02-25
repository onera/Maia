#pragma once


#include "std_e/parallel/serialize.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "maia/utils/parallel/exchange/global_numbering.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "pdm_block_to_part.h"
#include <algorithm>
#include <mpi.h>


namespace pdm {


class block_to_parts_protocol {
  public:
    block_to_parts_protocol(MPI_Comm mpi_comm, distribution_vector<PDM_g_num_t> dist, global_numberings lgs)
      : distribution(move(dist))
      , LN_to_GNs(move(lgs))
      , n_part(LN_to_GNs.size())
      , n_elts(n_part)
      , g_num_by_block(n_part)
    {
      // for (auto& LN_to_GN : LN_to_GNs) {
      //   std_e::offset(LN_to_GN,1); // TODO why does Paradigma uses 1-indexed LN_to_GN but 0-indexed distribution ???
      // }
      std::transform(begin(LN_to_GNs),end(LN_to_GNs),begin(n_elts),[](auto& LN_to_GN){ return LN_to_GN.size(); });
      std::transform(begin(LN_to_GNs),end(LN_to_GNs),begin(g_num_by_block),[](auto& LN_to_GN){ return LN_to_GN.data(); });

      PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&mpi_comm);

      btp = PDM_block_to_part_create(distribution.data(),
                                     (const PDM_g_num_t **) g_num_by_block.data(),
                                     n_elts.data(),
                                     n_part,
                                     pdm_comm);
    }

    std::vector<std_e::serialized_array> exchange(std_e::serialized_array& array_block) {
      auto strides = interval_lengths(array_block.offsets);
      int* blk_strid = strides.data();
      std::byte* blk_data = array_block.data.data();

      int**  part_stride;
      std::byte** part_data;
      PDM_block_to_part_exch(btp,
                             sizeof(std::byte),
                             PDM_STRIDE_VAR_INTERLACED,
                             blk_strid,
                             blk_data,
                            &part_stride,
                 (void ***) &part_data);

      // TODO Here, copying to std::vectors ONLY to replace manual calls to free by automatic RAII,
      // Should change Paradigm interface
      std::vector<std_e::serialized_array> res(n_part);

      for (int i=0; i<n_part; ++i) {
        int part_size = n_elts[i];
        res[i].offsets = indices_from_strides(std_e::make_span(part_stride[i],part_size));

        // TODO paradigm function take offsets instead of sizes?
        int part_cat_size = res[i].offsets.back();
        //for (int sz : res[i].sizes) {
        //  part_cat_size += sz;
        //}
        res[i].data = std::vector<std::byte>(part_cat_size);
        std::copy_n(part_data[i],part_cat_size,begin(res[i].data));

        free(part_stride[i]);
        free(part_data[i]);
      }
      free(part_stride);
      free(part_data);

      return res;
    }

    ~block_to_parts_protocol() {
      //PDM_block_to_part_free(btp);
    }

    // Don't know how PDM_block_to_part_t reacts to address changes
    // So prevent these unsafe operations
    block_to_parts_protocol(block_to_parts_protocol&&) = delete;
    block_to_parts_protocol& operator=(block_to_parts_protocol&&) = delete;
    block_to_parts_protocol(const block_to_parts_protocol&) = delete;
    block_to_parts_protocol& operator=(const block_to_parts_protocol&) = delete;

  private:
    distribution_vector<PDM_g_num_t> distribution;
    global_numberings LN_to_GNs;

    int n_part;
    std::vector<int> n_elts;
    std::vector<PDM_g_num_t*> g_num_by_block;

    PDM_block_to_part_t* btp;
};


class block_to_part_protocol {
  public:
    block_to_part_protocol(MPI_Comm mpi_comm, distribution_vector<PDM_g_num_t> dist, global_numbering lg)
      : impl(mpi_comm,std::move(dist),{std::move(lg)})
    {}
    template<
      class Global_numbering,
      std::enable_if_t<!std::is_same_v<std::decay_t<Global_numbering>,global_numbering>,int> =0
    >
    block_to_part_protocol(MPI_Comm mpi_comm, distribution_vector<PDM_g_num_t> dist, Global_numbering lg)
      : block_to_part_protocol(mpi_comm,dist,global_numbering(begin(lg),end(lg)))
    {}

    std_e::serialized_array exchange(std_e::serialized_array& array) {
      auto res = impl.exchange(array);
      assert(res.size()==1);
      return res[0];
    }
  private:
    block_to_parts_protocol impl;
};




template<class T> auto
exchange(block_to_part_protocol& btp, const std::vector<T>& dist_array) -> std::vector<T> {
  auto dist_serial = std_e::serialize_array(dist_array);
  auto part_serial =  btp.exchange(dist_serial);
  return std_e::deserialize_array<T>(part_serial);
}
template<class T> auto
exchange(block_to_parts_protocol& btp, const std::vector<T>& dist_array) -> std::vector<std::vector<T>> {
  auto dist_serial = std_e::serialize_array(dist_array);
  auto parts_serial =  btp.exchange(dist_serial);

  int n_parts = parts_serial.size();
  std::vector<std::vector<T>> res(n_parts);
  for (int i=0; i<n_parts; ++i) {
    res[i] = std_e::deserialize_array<T>(parts_serial[i]);
  }
  return res;
}


} // pdm
