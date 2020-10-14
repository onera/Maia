#ifndef _MAIA_UTILS_MPI_SCHEDULER_
#define _MAIA_UTILS_MPI_SCHEDULER_

#include <mpi.h>
#include <vector>
#include <functional>


void run_scheduler(MPI_Comm&                                    comm,
                   std::vector<int>&                            n_rank_for_test,
                   std::vector<std::function<void(MPI_Comm&)>>& tests_suite);

#endif /* _MAIA_UTILS_MPI_SCHEDULER_ */

