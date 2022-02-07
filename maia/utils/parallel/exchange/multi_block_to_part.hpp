#pragma once


#include "pdm_multi_block_to_part.h"
#include "std_e/future/contract.hpp"
#include <vector>
#include <numeric>


namespace pdm {


// NOTE: this wrapper only handles constant stride of 1, and n_part=1
class multi_block_to_part_protocol {
  public:
    template<class Range_of_distributions, class g_num_range>
    multi_block_to_part_protocol(const Range_of_distributions& distribs, const g_num_range& ln_to_gn_0, MPI_Comm comm)
    {
      PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&comm);

      ln_to_gn = std::vector<const PDM_g_num_t*> {ln_to_gn_0.data()};

      n_block = distribs.size();

      int n_elts_0 = ln_to_gn_0.size();
      n_elts = std::vector<int>{n_elts_0};

      block_distribs = std::vector<const PDM_g_num_t*>(n_block);
      for (int i=0; i<n_block; ++i) {
        block_distribs[i] = distribs[i].data();
      }

      multi_distrib_idx = std::vector<PDM_g_num_t>(n_block+1);
      multi_distrib_idx[0] = 0;
      for (int i=0; i<n_block; ++i) {
        multi_distrib_idx[i+1] = multi_distrib_idx[i] + distribs[i].back();
      }

      mbtp = PDM_multi_block_to_part_create(
                              multi_distrib_idx.data(),
                              n_block,
        (const PDM_g_num_t**) block_distribs.data(),
        (const PDM_g_num_t**) ln_to_gn.data(),
                              n_elts.data(),
                              n_part,
                              pdm_comm
      );
    }

    template<class Range_of_ranges> auto
    exchange(const Range_of_ranges& d_arrays) {
      using range_type = typename Range_of_ranges::value_type;
      using T = typename range_type::value_type;
      STD_E_ASSERT(d_arrays.size()==n_block);
      // prepare d_array_ptr
      std::vector<const T*> darray_ptr(n_block);
      for (int i=0; i<n_block; ++i) {
        darray_ptr[i] = d_arrays[i].data();
      }

      // stride
      int stride = 1;
      int** stride_one = (int ** ) malloc( n_block * sizeof(int *));
      for(int i_block = 0; i_block < n_block; ++i_block){
        stride_one[i_block] = (int * ) malloc( 1 * sizeof(int));
        stride_one[i_block][0] = stride;
      }

      // exchange
      T** parray = nullptr;
      PDM_multi_block_to_part_exch2(
                   mbtp, sizeof(T), PDM_STRIDE_CST_INTERLACED,
                   stride_one,
        (void ** ) darray_ptr.data(),
                   nullptr,
        (void ***) &parray
      );

      // free strides
      for(int i_block = 0; i_block < n_block; ++i_block){
        free(stride_one[i_block]);
      }
      free(stride_one);

      // extract data
      std::vector<T> p_res(n_elts[0]);
      for (int i=0; i<n_elts[0]; ++i) {
        p_res[i] = parray[0][i];
      }

      // free data
      for(int i_part = 0; i_part < n_part; ++i_part){
        free(parray[i_part]);
      }
      free(parray);

      return p_res;
    }


    ~multi_block_to_part_protocol() {
      PDM_multi_block_to_part_free(mbtp);
    }

  private:
    std::vector<const PDM_g_num_t*> ln_to_gn;
    std::vector<const PDM_g_num_t*> block_distribs;
    const int n_part = 1;
    int n_block;
    std::vector<int> n_elts;
    std::vector<PDM_g_num_t> multi_distrib_idx;
    PDM_multi_block_to_part_t* mbtp;
};



} // pdm
