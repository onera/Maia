#include "doctest/extensions/doctest_mpi.h"
#include "maia/utils/parallel/exchange/multi_block_to_part.hpp"


MPI_TEST_CASE("[3p] multi_block_to_part n_block=2",3) {
  int i_rank = test_rank;

  // what we have
  int n_block = 2;
  std::vector<std::vector<PDM_g_num_t>> distribs = {{0,7,14,21},{0,7,14,21}};
  std::vector<std::vector<int32_t>> d_arrays;
  if (i_rank==0) d_arrays  = { { 0, 1, 2, 3, 4, 5, 6} , {  0, 10, 20, 30, 40, 50, 60} };
  if (i_rank==1) d_arrays  = { { 7, 8, 9,10,11,12,13} , { 70, 80, 90,100,110,120,130} };
  if (i_rank==2) d_arrays  = { {14,15,16,17,18,19,20} , {140,150,160,170,180,190,200} };

  // how we want to exchange data (here, we want to concatenate the two distributed array into one)
  std::vector<PDM_g_num_t> merged_distri = {0,14,28,42};

  int n_elts = merged_distri[i_rank+1]-merged_distri[i_rank];
  std::vector<PDM_g_num_t> ln_to_gn(n_elts);
  std::iota(begin(ln_to_gn),end(ln_to_gn),merged_distri[i_rank]+1);

  // exchange
  pdm::multi_block_to_part_protocol mbtp(distribs,ln_to_gn,test_comm);
  auto p_data = mbtp.exchange(d_arrays);

  // check
  std::vector<int32_t> parray_expected;
  if (i_rank==0) parray_expected = {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13};
  if (i_rank==1) parray_expected = { 14, 15, 16, 17, 18, 19, 20,  0, 10, 20, 30, 40, 50, 60};
  if (i_rank==2) parray_expected = { 70, 80, 90,100,110,120,130,140,150,160,170,180,190,200};
  MPI_CHECK(0, p_data == parray_expected);
  MPI_CHECK(1, p_data == parray_expected);
  MPI_CHECK(2, p_data == parray_expected);
}
