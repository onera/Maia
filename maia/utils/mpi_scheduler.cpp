#include <cassert>
#include <unistd.h>
#include "std_e/algorithm/distribution.hpp"
#include "maia/utils/mpi_scheduler.hpp"
#include "std_e/logging/log.hpp"
#include "std_e/interval/knot_sequence.hpp"


// --------------------------------------------------------------------------------------
double big_loop(){

  int n_iteration = 1000;
  int size = 100000000;
  std::vector<double> test(size, -1.);
  for(int iteration = 0; iteration < n_iteration; ++iteration){
    for(int i = 0; i < size; ++i){
      test[i] += 2.*iteration/((i+1)*(i+1));
    }
    test[100] = iteration;
    sleep(0.1);
    printf("iteration : %i \n ", iteration);
    std::cout << " iteration : " << iteration << std::endl;
  }

  return test[0];

}

// --------------------------------------------------------------------------------------
void update_window_to_math_test(std::vector<int>& dtest_proc,
                                std::vector<int>& n_rank_for_test,
                                int&              i_test_g,
                                std::vector<int>& remote,
                                MPI_Win&          win,
                                int               i_rank)
{
  bool run_this_test = false;
  int i_target_rank = std_e::interval_index(i_test_g, dtest_proc);
  printf("[%i] Going to lock - MPI_Win_lock \n", i_rank);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win);
  printf("[%i] Going to lock - MPI_Win_lock end \n", i_rank);

  // Compute relative index in remote window iloc = iglob - shift
  int i_test_loc = i_test_g - dtest_proc[i_target_rank];

  MPI_Get(remote.data(),    /* origin_addr     */
          1,                /* origin_count    */
          MPI_INT,          /* origin_datatype */
          i_target_rank,    /* target_rank     */
          i_test_loc,       /* target_disp     */
          1,                /* target_count    */
          MPI_INT,          /* target_datatype */
          win);             /* win             */

  // This one seem to be neccessary
  MPI_Win_flush(i_target_rank, win);

  printf("[%i] have value : %i | i_target_rank = %i \n", i_rank, remote[0], i_target_rank);
  if(remote[0] < n_rank_for_test[i_test_g]){
    remote[0] += 1;
    printf("[%i] update with n_cur_test = %i \n", i_rank, remote[0]);
    MPI_Put(remote.data(),    /* origin_addr     */
            1,                /* origin_count    */
            MPI_INT,          /* origin_datatype */
            i_target_rank,    /* target_rank     */
            i_test_loc,       /* target_disp     */
            1,                /* target_count    */
            MPI_INT,          /* target_datatype */
            win);             /* win             */
    run_this_test = true;
  }

  MPI_Win_flush(i_target_rank, win);
  MPI_Win_unlock(i_target_rank, win);


}

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

  MPI_Group world_group;
  MPI_Comm_group(comm, &world_group);

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

  // MPI_Win_post(world_group, 0, win_count_rank_for_test);
  // MPI_Win_start(world_group, 0, win_count_rank_for_test);

  // III/ Setup
  std::vector<int> remote_copy_count_rank(2, -1); // []
  int i_test_g = 0;

  MPI_Barrier(comm);
  // if(i_rank == 0 ){
  //   auto res = big_loop();
  //   printf("res = %12.5e \n", res);
  // }

  // win_p0 : [ 0, 0 ] // win_p1 : [ 0 ]
  bool last_decision_failed = false; // 0 = false / 1 = true
  // if(i_rank == 0){
  //   last_decision_failed = true;
  // }
  bool run_this_test        = false;

  if(!last_decision_failed ){

    int i_target_rank = std_e::interval_index(i_test_g, dtest_proc);

    // --------------------------------------------------------------------------------
    printf("[%i] Going to lock - MPI_Win_lock \n", i_rank);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win_count_rank_for_test);
    printf("[%i] Going to lock - MPI_Win_lock end \n", i_rank);

    // Compute relative index in remote window iloc = iglob - shift
    int i_test_loc = i_test_g - dtest_proc[i_target_rank];

    MPI_Get(remote_copy_count_rank.data(),    /* origin_addr     */
            1,                                /* origin_count    */
            MPI_INT,                          /* origin_datatype */
            i_target_rank,                    /* target_rank     */
            i_test_loc,                       /* target_disp     */
            1,                                /* target_count    */
            MPI_INT,                          /* target_datatype */
            win_count_rank_for_test);         /* win             */

    // This one seem to be neccessary
    MPI_Win_flush(i_target_rank, win_count_rank_for_test);

    printf("[%i] have value : %i | i_target_rank = %i \n", i_rank, remote_copy_count_rank[0], i_target_rank);
    if(remote_copy_count_rank[0] < n_rank_for_test[i_test_g]){
      remote_copy_count_rank[0] += 1;
      printf("[%i] update with n_cur_test = %i \n", i_rank, remote_copy_count_rank[0]);
      MPI_Put(remote_copy_count_rank.data(),    /* origin_addr     */
              1,                                /* origin_count    */
              MPI_INT,                          /* origin_datatype */
              i_target_rank,                    /* target_rank     */
              i_test_loc,                       /* target_disp     */
              1,                                /* target_count    */
              MPI_INT,                          /* target_datatype */
              win_count_rank_for_test);         /* win             */
      run_this_test = true;
    }

    // MPI_Win_flush(i_target_rank, win_count_rank_for_test);
    MPI_Win_unlock(i_target_rank, win_count_rank_for_test);

    // --------------------------------------------------------------------------------
    remote_copy_count_rank[0] = -10;

    printf("[%i] Rank 0 go sleep \n", i_rank);
    // if(i_rank == 0 ){
    //   sleep(5);
    // } else if (i_rank == 1) {
    //   sleep(6);
    // }

    // --------------------------------------------------------------------------------
    // bool is_ready_with_other = false;
    // while( !is_ready_with_other && run_this_test ){

    //   printf("[%i] On n'attends pas Patrik - MPI_Win_lock \n", i_rank);
    //   MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win_count_rank_for_test);
    //   printf("[%i] On n'attends pas Patrik - MPI_Win_lock end \n", i_rank);

    //   // Compute relative index in remote window iloc = iglob - shift
    //   int i_test_loc = i_test_g - dtest_proc[i_target_rank];

    //   MPI_Get(remote_copy_count_rank.data(),    /* origin_addr     */
    //           1,                                /* origin_count    */
    //           MPI_INT,                          /* origin_datatype */
    //           i_target_rank,                    /* target_rank     */
    //           i_test_loc,                       /* target_disp     */
    //           1,                                /* target_count    */
    //           MPI_INT,                          /* target_datatype */
    //           win_count_rank_for_test);         /* win             */

    //   // This one seem to be neccessary
    //   MPI_Win_flush(i_target_rank, win_count_rank_for_test);
    //   MPI_Win_unlock(i_target_rank, win_count_rank_for_test);
    //   is_ready_with_other = remote_copy_count_rank[0] == n_rank_for_test[i_test_g];
    //   printf("[%i] I'm waiting my bro is_ready_with_other :: %i | run_this_test :: %i | remote_copy_count_rank :: %i \n", i_rank, is_ready_with_other, run_this_test, remote_copy_count_rank[0]);
    // }
    // --------------------------------------------------------------------------------


    // --------------------------------------------------------------------------------

    if(run_this_test) {

      // > Create group
      // assert(group_size == n_rank_for_test[i_test_g])

    } else {
      i_test_g++;
    }
  }

  // MPI_Win_complete(win_count_rank_for_test);
  // MPI_Win_wait(win_count_rank_for_test);
  // MPI_Barrier(comm);

  // Free
  MPI_Win_free(&win_global_n_rank_available);
  MPI_Win_free(&win_count_rank_for_test);
}
