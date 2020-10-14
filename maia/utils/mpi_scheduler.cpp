#include <cassert>
#include "maia/utils/mpi_scheduler.hpp"




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










}
