#ifndef _MAIA_UTILS_MPI_SCHEDULER_
#define _MAIA_UTILS_MPI_SCHEDULER_

#include <mpi.h>
#include <vector>
#include <functional>

enum class window_state {
  WORKING,
  MODIFYING,
  DECISION_MAKING
};



void run_scheduler(MPI_Comm&                                    comm,
                   std::vector<int>&                            n_rank_for_test,
                   std::vector<std::function<void(MPI_Comm&)>>& tests_suite);
void run_scheduler_old(MPI_Comm&                                    comm,
                       std::vector<int>&                            n_rank_for_test,
                       std::vector<std::function<void(MPI_Comm&)>>& tests_suite);

bool
update_window_to_match_test(std::vector<int>& dtest_proc,
                            std::vector<int>& n_rank_for_test,
                            std::vector<int>& list_rank_for_test_idx,
                            int&              i_test_g,
                            std::vector<int>& remote,
                            MPI_Win&          win,
                            MPI_Win&          win_list_rank,
                            int               i_rank);
void
update_list_rank_for_test(std::vector<int>& dtest_proc,
                          std::vector<int>& n_rank_for_test,
                          std::vector<int>& list_rank_for_test,
                          std::vector<int>& list_rank_for_test_idx,
                          int&              i_test_g,
                          int&              i_target_rank,
                          MPI_Win&          win_list_rank,
                          int               i_rank);


// void
// update_list_rank_for_test(std::vector<int>& dtest_proc,
//                           std::vector<int>& n_rank_for_test,
//                           int*              list_rank_for_test,
//                           std::vector<int>& list_rank_for_test_idx,
//                           int&              i_test_g,
//                           int&              i_target_rank,
//                           MPI_Win&          win_list_rank,
//                           int               i_rank);

std::vector<int>
setup_test_distribution(MPI_Comm&         comm,
                        std::vector<int>& n_rank_for_test);

#endif /* _MAIA_UTILS_MPI_SCHEDULER_ */

