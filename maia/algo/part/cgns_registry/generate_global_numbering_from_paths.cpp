#include "maia/algo/part/cgns_registry/generate_global_numbering_from_paths.hpp"

#include <algorithm>
#include <string>
#include <numeric>
#include <cassert>
#include <iostream>
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/interval/interval_sequence.hpp"
#include "std_e/parallel/mpi.hpp"
// #include "logging/logging.hpp"
#include "std_e/utils/vector.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/logging/log.hpp"
// #include "pdm_sort.h"
// #include "pdm_gnum_from_hash_values.h"
// ------------------------------------------------------------------
//

// ===========================================================================
std::vector<PDM_g_num_t> generate_global_id(      int                           n_loc_id,
                                            const std::vector<std::string>&     block_paths,
                                            const std::vector<int>&             send_n,
                                            const std_e::interval_vector<int>&  send_idx,
                                            const std::vector<int>&             recv_n,
                                            const std_e::interval_vector<int>&  recv_idx,
                                                  MPI_Comm                      comm)
{
  // -------------------------------------------------------------------
  // 1 - Identify an order without touching the original ordering ---> Preserve send/recv
  int n_block = block_paths.size();
  std::vector<int> order_name(n_block);
  std::iota(begin(order_name), end(order_name), 0);
  std::sort(begin(order_name), end(order_name), [&](const int& i1, const int& i2){
    if(std::hash<std::string>{}(block_paths[i1]) == std::hash<std::string>{}(block_paths[i2])){
      return block_paths[i1] < block_paths[i2];
    } else {
      return std::hash<std::string>{}(block_paths[i1]) < std::hash<std::string>{}(block_paths[i2]);
    }
  });

  // if(n_block > 0) {
  //   printf("n_block = %i \n", n_block);
  // }

  // -------------------------------------------------------------------
  // 2 - Give an local number for each element in block_paths
  std::vector<PDM_g_num_t> global_name_num(n_block);
  PDM_g_num_t next_name_id =  0;
  PDM_g_num_t n_loc_name_id =  0;
  std::string lastName;
  for(int i = 0; i < n_block; ++i){
    if(block_paths[order_name[i]] == lastName){
      global_name_num[order_name[i]] = next_name_id;
    } else {
      next_name_id++;
      n_loc_name_id++;
      global_name_num[order_name[i]] = next_name_id;
      lastName = block_paths[order_name[i]];
    }
  }

  // -------------------------------------------------------------------
  // 3 - Setup global numbering by simply shift
  PDM_g_num_t shift_g;
  PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&comm);
  int ierr = PDM_MPI_Scan(&n_loc_name_id, &shift_g, 1, PDM__PDM_MPI_G_NUM, PDM_MPI_SUM, pdm_comm);
  assert(ierr == 0);
  shift_g -= n_loc_name_id;

  for(int i = 0; i < n_block; ++i){
    global_name_num[i] += shift_g;
  }

  // Panic verbose
  // for(int i = 0; i < n_block; ++i) {
  //   std::string s;
  //   s += block_paths[i] + " --> " + std::to_string(global_name_num[i]) + " - order= " + std::to_string(order_name[i]) + " \n";
  //   std_e::log("sonics", s);
  // }

  // -------------------------------------------------------------------
  // 4 - Back to original distribution
  std::vector<PDM_g_num_t> part_global_id(n_loc_id);
  PDM_MPI_Alltoallv(global_name_num.data(), (int *) recv_n.data(), (int *) recv_idx.data(), PDM__PDM_MPI_G_NUM,
                    part_global_id.data() , (int *) send_n.data(), (int *) send_idx.data(), PDM__PDM_MPI_G_NUM, pdm_comm);

  //std_e::offset(part_global_id,-1);
  return part_global_id;
}

// ===========================================================================
// On cherche a creer une numerotation absolue plutot...
std::vector<PDM_g_num_t> generate_global_numbering(/* TODO const */ std::vector<std::string>& part_paths,
                                                   MPI_Comm                  comm)
{
  int n_rank = std_e::n_rank(comm);

  // -------------------------------------------------------------------
  std_e::sort_unique(part_paths);
  int n_loc_id = part_paths.size();

  // -------------------------------------------------------------------
  // 1 - Generate keys from path
  // DOC we use a hash for two reasons:
  // - we need a fast way to generate a reasonably balanced distribution of all unique path into slots (here, the mpi ranks)
  //    since hash collisions are rare, chances of many different paths having the same hash, and hence being distributed on the same slot, is also rare
  // - equal paths have equal hashes, so they will end up in the same mpi rank, so generating a unique global id is then local to the rank
  //    we don't mind that different paths may have the same hash: they will be distributed on the same slot, but then they are treated separatly
  std::vector<size_t> part_paths_code = std_e::hash_vector(part_paths);

  // for(int i = 0; i < part_paths_code.size(); ++i) {
  //   std::string s;
  //   s += "part_paths = " + part_paths[i] + " --> " + std::to_string(part_paths_code[i]) + " \n";
  //   std_e::log("sonics", s);
  // }

  // -------------------------------------------------------------------
  // 2 - Generate distribution
  // DOC we make the hypothesis that the hashes repartition will be almost uniform in [0,2^64)
  // TODO Repartion from sampling (
  distribution_vector<size_t> distrib_key = uniform_distribution(n_rank, std::numeric_limits<size_t>::max());

  //// -------------------------------------------------------------------
  //// 3 - TODO Not used -- Store for each string their size in order to make strideBuffer on char
  //std::vector<int> part_pathsSize(n_loc_id);
  //for(int i = 0; i < n_loc_id; ++i){
  //  part_pathsSize[i] = static_cast<int>(part_paths[i].size());
  //}

  // -------------------------------------------------------------------
  // 4 - Prepare buffer send
  std::vector<int> send_n(n_rank);
  std::vector<int> recv_n(n_rank);
  std::vector<int> send_str_n(n_rank);
  std::vector<int> recv_str_n(n_rank);
  std::fill(begin(send_n    ), end(send_n    ), 0);
  std::fill(begin(send_str_n), end(send_str_n), 0);

  assert(static_cast<int>(part_paths    .size()) == n_loc_id);
  assert(static_cast<int>(part_paths_code.size()) == n_loc_id);

  for(int i = 0; i < n_loc_id; ++i){
    // std::cout << "part_paths_code[" << i << "/" << part_paths_code.size() << "]" << distrib_key.size() << std::endl;
    int i_rank = search_rank(part_paths_code[i], distrib_key);
    assert(i_rank >= 0);
    int n_data_to_send = sizeof(char) * part_paths[i].size();
    send_str_n[i_rank] += n_data_to_send;
    send_n[i_rank]++;
  }

  // -------------------------------------------------------------------
  // Panic verbose
  // for(int i = 0; i < n_rank; ++i){
  //   e_log << "send_n   [" << i << "] = " << send_n[i]    << std::endl;
  //   e_log << "send_str_n[" << i << "] = " << send_str_n[i] << std::endl;
  // }
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // 5 - Exchange
  MPI_Alltoall(send_n    .data(), 1, MPI_INT, recv_n    .data(), 1, MPI_INT, comm);
  MPI_Alltoall(send_str_n.data(), 1, MPI_INT, recv_str_n.data(), 1, MPI_INT, comm);

  // -------------------------------------------------------------------
  // Panic verbose
  // for(int i = 0; i < n_rank; ++i){
  //   e_log << "recv_n   [" << i << "] = " << recv_n[i]    << std::endl;
  //   e_log << "recv_str_n[" << i << "] = " << recv_str_n[i] << std::endl;
  // }
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // 6 - Compute all index (need for MPI and algorithm)
  std_e::interval_vector<int> send_idx     = std_e::indices_from_strides(send_n    );
  std_e::interval_vector<int> recv_idx     = std_e::indices_from_strides(recv_n    );
  std_e::interval_vector<int> send_str_idx = std_e::indices_from_strides(send_str_n);
  std_e::interval_vector<int> recv_str_idx = std_e::indices_from_strides(recv_str_n);

  // -------------------------------------------------------------------
  // 7 - Allocation of buffer
  std::vector<int>  send_buffer    (send_idx    .back());
  std::vector<int>  recv_buffer    (recv_idx    .back());
  std::vector<char> send_str_buffer(send_str_idx.back());
  std::vector<char> recv_str_buffer(recv_str_idx.back());

  // -------------------------------------------------------------------
  // 8 - Fill buffer send // TODO serialize?
  std::vector<int> send_count   (n_rank);
  std::vector<int> send_str_count(n_rank);
  std::fill(begin(send_count    ), end(send_count    ), 0);
  std::fill(begin(send_str_count), end(send_str_count), 0);
  for(int i = 0; i < n_loc_id; ++i){
    int i_rank_to_send = search_rank(part_paths_code[i], distrib_key);
    int string_size  = part_paths[i].size();
    send_buffer[send_idx[i_rank_to_send]+send_count[i_rank_to_send]++] = string_size;
    for(int j = 0; j < string_size; ++j){
      send_str_buffer[send_str_idx[i_rank_to_send]+send_str_count[i_rank_to_send]++] = part_paths[i][j];
    }
  }

  // -------------------------------------------------------------------
  // 9 - Exchange
  MPI_Alltoallv(send_buffer.data(), send_n.data(), send_idx.data(), MPI_INT,
                recv_buffer.data(), recv_n.data(), recv_idx.data(), MPI_INT, comm);
  MPI_Alltoallv(send_str_buffer.data(), send_str_n.data(), send_str_idx.data(), MPI_BYTE,
                recv_str_buffer.data(), recv_str_n.data(), recv_str_idx.data(), MPI_BYTE, comm);

  // -------------------------------------------------------------------
  // 10 - Post-treat exchange and rebuild string
  int n_string_to_recv = std::accumulate(begin(recv_n), end(recv_n), 0);
  std::vector<std::string> block_paths(n_string_to_recv);
  int idxG = 0;
  for(int i = 0; i < n_rank; ++i){
    int beg_recv     = recv_idx    [i];
    int beg_recv_str = recv_str_idx[i];
    // e_log << " Recv from " << i << " at " << beg_recv << std::endl;
    // for(int idx_data = 0; idx_data < recv_n[i]; idx_data++){
    //   e_log << " Recv_buffer[" << beg_recv+idx_data << "] = " << recv_buffer[beg_recv+idx_data] << std::endl;
    // }
    int idx_str = 0;
    for(int idx_data = 0; idx_data < recv_n[i]; idx_data++){
      int size_name_recv = recv_buffer[beg_recv+idx_data];
      for(int j = 0; j < size_name_recv; ++j){
        block_paths[idxG] += recv_str_buffer[beg_recv_str+idx_str++];
      }
      // e_log << "block_paths[" << idxG << "] = " << block_paths[idxG] << std::endl;
      idxG++;
    }
  }

  // -------------------------------------------------------------------
  // 11 - Order
  std::vector<PDM_g_num_t> recv_global_id = generate_global_id(n_loc_id, block_paths, send_n, send_idx, recv_n, recv_idx, comm);

  // Revert buffer
  std::vector<PDM_g_num_t> part_global_id(n_loc_id);
  std::fill(begin(send_n    ), end(send_n    ), 0);
  for(int i = 0; i < n_loc_id; ++i){
    int i_rank = search_rank(part_paths_code[i], distrib_key);
    assert(i_rank >= 0);

    int idx_read = send_idx[i_rank] + send_n[i_rank]++;
    part_global_id[i] = recv_global_id[idx_read];
  }

  return part_global_id;
}
