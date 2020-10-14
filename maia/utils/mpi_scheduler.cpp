#include <cassert>
#include "std_e/algorithm/distribution.hpp"
#include "maia/utils/mpi_scheduler.hpp"
#include "std_e/logging/log.hpp"
#include "std_e/interval/knot_sequence.hpp"

// --------------------------------------------------------------------------------------
std::vector<int>
setup_test_distribution(MPI_Comm&         comm,
                        std::vector<int>& n_rank_for_test)
{
  int n_rank;
  MPI_Comm_size(comm, &n_rank);
  std::vector<int> dtest_proc(1+n_rank);
  std_e::uniform_distribution(begin(dtest_proc), end(dtest_proc), static_cast<int>(n_rank_for_test.size()));

  std::string s = "dtest_proc::";
  for(int i = 0; i < n_rank+1; ++i){
    s += std::to_string(dtest_proc[i]) + " ";
  }
  s += "\n";
  std::cout << s << std::endl;
  // std_e::log("file", s);

  // return std_e::to_knot_vector<int>(dtest_proc);
  return dtest_proc;
}

// --------------------------------------------------------------------------------------
void run_scheduler(MPI_Comm&                                    comm,
                   std::vector<int>&                            n_rank_for_test,
                   std::vector<std::function<void(MPI_Comm&)>>& tests_suite)
{

  // First we setup the main communicator parameter
  int i_rank, n_rank;
  MPI_Comm_size(comm, &n_rank);
  MPI_Comm_rank(comm, &i_rank);

  // Hypothesis : all rank have the same list test and organize in a same way
  int n_tot_test = n_rank_for_test.size();
  assert(n_tot_test == static_cast<int>(tests_suite.size()));

  // Well we need multiple window to organize the algorithm
  // I/ Setup the global variable containing the number of rank available
  MPI_Win win_global_n_rank_available;
  std::vector<int> global_n_rank_available;
  if(i_rank == 0) {
    global_n_rank_available.resize(3);
    global_n_rank_available[0] = n_rank;
    global_n_rank_available[1] = static_cast<int>(window_state::WORKING);
    global_n_rank_available[2] = -1; /* On sait jamais */
  }

  MPI_Win_create(global_n_rank_available.data(),              /* base ptr */
                 global_n_rank_available.size()*sizeof(int),  /* size     */
                 sizeof(int),                                 /* disp_unit*/
                 MPI_INFO_NULL,
                 comm,
                 &win_global_n_rank_available);

  // Input : (parsing pytest mark)
  // dtest_proc   : n_proc (distribution des test : par exemple : [0 2 4]) si n_test = 2 pour 0 et 1
  // dtest_n_proc : dn_test              proc 0  [ 1 2 ] et proc 1 : [ 2 1 ]


  // II/ Setup the distributed rank value for each test
  auto dtest_proc = setup_test_distribution(comm, n_rank_for_test);

  // For each test we store a distributed array that contains the number of rank that can execute the test
  // Increment by one via MPI_Put
  std::vector<int> count_rank_for_test(dtest_proc[i_rank+1] - dtest_proc[i_rank], 0);
  // std::vector<int> list_rank_for_test(dtest_proc[i_rank+1] - dtest_proc[i_rank], 0);

  MPI_Win win_count_rank_for_test;
  MPI_Win_create(count_rank_for_test.data(),              /* base ptr */
                 count_rank_for_test.size()*sizeof(int),  /* size     */
                 sizeof(int),                             /* disp_unit*/
                 MPI_INFO_NULL,
                 comm,
                 &win_count_rank_for_test);
  MPI_Barrier(comm);
  // III/ Setup
  std::vector<int> remote_copy_count_rank(2); // []
  int i_test = 0;

  bool last_decision_failed = false; // 0 = false / 1 = true
  if(!last_decision_failed ){

    int i_target_rank = std_e::interval_index(i_test, dtest_proc);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win_count_rank_for_test);
    MPI_Get(remote_copy_count_rank.data(),    /* origin_addr     */
            1,                                /* origin_count    */
            MPI_INT,                          /* origin_datatype */
            i_target_rank,                    /* target_rank     */
            0,                                /* target_disp     */
            1,                                /* target_count    */
            MPI_INT,                          /* target_datatype */
            win_count_rank_for_test);         /* win             */
    printf("[%i] have value : %i | i_target_rank = %i \n", i_rank, remote_copy_count_rank[0], i_target_rank);
    if(remote_copy_count_rank[0] < n_rank_for_test[i_test]){
      remote_copy_count_rank[0] += 1;
      MPI_Put(remote_copy_count_rank.data(),    /* origin_addr     */
              1,                                /* origin_count    */
              MPI_INT,                          /* origin_datatype */
              i_target_rank,                    /* target_rank     */
              0,                                /* target_disp     */
              1,                                /* target_count    */
              MPI_INT,                          /* target_datatype */
              win_count_rank_for_test);         /* win             */
    }
    MPI_Win_unlock(i_target_rank, win_count_rank_for_test);
  }

  // Free
  MPI_Win_free(&win_global_n_rank_available);
  MPI_Win_free(&win_count_rank_for_test);
}
