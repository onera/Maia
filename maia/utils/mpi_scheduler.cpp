#include <cassert>
#include "std_e/algorithm/distribution.hpp"
#include "maia/utils/mpi_scheduler.hpp"

// --------------------------------------------------------------------------------------
std::vector<int>
setup_test_distribution(MPI_Comm&         comm,
                        std::vector<int>& n_rank_for_test)
{
  int n_rank;
  MPI_Comm_size(comm, &n_rank);
  std::vector<int> dtest_proc(1+n_rank);
  std_e::uniform_distribution(begin(dtest_proc), end(dtest_proc), static_cast<int>(n_rank_for_test.size()));

  std::cout << "dtest_proc::";
  for(int i = 0; i < n_rank+1; ++i){
    std::cout << dtest_proc[i] << " ";
  }
  std::cout << std::endl;

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

  // III/ Setup
  bool last_decision_failed = false; // 0 = false / 1 = true
  if(!last_decision_failed ){


  }

  // Free
  MPI_Win_free(&win_global_n_rank_available);
}
