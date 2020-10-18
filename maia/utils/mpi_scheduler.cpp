#include <cassert>
#include <unistd.h>
#include <future>
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
bool
update_window_to_match_test(std::vector<int>& dtest_proc,
                            std::vector<int>& n_rank_for_test,
                            std::vector<int>& list_rank_for_test_idx,
                            int&              i_test_g,
                            std::vector<int>& remote,
                            MPI_Win&          win,
                            MPI_Win&          win_list_rank,
                            int               i_rank)
{
  bool run_this_test = false;

  int i_target_rank = std_e::interval_index(i_test_g, dtest_proc);
  // printf("[%i] update_window_to_match_test - MPI_Win_lock  | i_test_g:: %i   \n", i_rank, i_test_g);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win);
  // printf("[%i] update_window_to_match_test - MPI_Win_lock end | i_test_g:: %i \n", i_rank, i_test_g);

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

  // printf("[%i] have value : %i | i_target_rank = %i | i_test_g : %i  \n", i_rank, remote[0], i_target_rank, i_test_g);
  if(remote[0] < n_rank_for_test[i_test_g]){
    remote[0] += 1;
    // printf("[%i] update with n_cur_test = %i | i_test_g : %i \n", i_rank, remote[0], i_test_g);
    MPI_Put(remote.data(),    /* origin_addr     */
            1,                /* origin_count    */
            MPI_INT,          /* origin_datatype */
            i_target_rank,    /* target_rank     */
            i_test_loc,       /* target_disp     */
            1,                /* target_count    */
            MPI_INT,          /* target_datatype */
            win);             /* win             */
    run_this_test = true;

    MPI_Win_flush(i_target_rank, win);

    // Well we know that the state is lock we take advantage of this to setup the rank list
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win_list_rank);

    int beg_list_rank = list_rank_for_test_idx[i_test_g] - list_rank_for_test_idx[dtest_proc[i_target_rank]];
    int idx_list_rank = beg_list_rank + remote[0] - 1; // Because increment by one at the beginning

    // printf("[%i] update idx_list_rank at = %i with val = %i \n", i_rank, idx_list_rank, i_rank);
    MPI_Put(&i_rank,          /* origin_addr     */
            1,                /* origin_count    */
            MPI_INT,          /* origin_datatype */
            i_target_rank,    /* target_rank     */
            idx_list_rank,    /* target_disp     */
            1,                /* target_count    */
            MPI_INT,          /* target_datatype */
            win_list_rank);   /* win             */
    MPI_Win_flush(i_target_rank, win_list_rank);
    MPI_Win_unlock(i_target_rank, win_list_rank);

    // if(remote[0] == n_rank_for_test[i_test_g]){
      // printf(" Flush all \n");
    // }

  }

  // MPI_Win_flush_all(win);
  // MPI_Win_flush_local_all(win);
  MPI_Win_unlock(i_target_rank, win);

  return run_this_test;
}

// --------------------------------------------------------------------------------------
void
update_list_rank_for_test(std::vector<int>& dtest_proc,
                          std::vector<int>& n_rank_for_test,
                          std::vector<int>& list_rank_for_test,
                          std::vector<int>& list_rank_for_test_idx,
                          int&              i_test_g,
                          int&              i_target_rank,
                          MPI_Win&          win_list_rank,
                          int               i_rank)
{
  // printf("[%i] update_list_rank_for_test - MPI_Win_lock \n", i_rank);
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, i_target_rank, 0, win_list_rank);
  // printf("[%i] update_list_rank_for_test - MPI_Win_lock end \n", i_rank);

  // Compute relative index in remote window iloc = iglob - shift
  int beg_cur_test  = list_rank_for_test_idx[i_test_g];
  int beg_list_rank = list_rank_for_test_idx[i_test_g] - list_rank_for_test_idx[dtest_proc[i_target_rank]];

  // printf("[%i] update_list_rank_for_test : beg_cur_test  %i / %i \n", i_rank, beg_cur_test, list_rank_for_test.size());
  // printf("[%i] update_list_rank_for_test : beg_list_rank %i \n", i_rank, beg_list_rank);
  // printf("[%i] run_this_test : %i \n", i_rank, list_rank_for_test[beg_cur_test]);
  MPI_Get(&list_rank_for_test[beg_cur_test],     /* origin_addr     */
          n_rank_for_test[i_test_g],             /* origin_count    */
          MPI_INT,                               /* origin_datatype */
          i_target_rank,                         /* target_rank     */
          beg_list_rank,                         /* target_disp     */
          n_rank_for_test[i_test_g],             /* target_count    */
          MPI_INT,                               /* target_datatype */
          win_list_rank);                        /* win             */

  // This one seem to be neccessary
  MPI_Win_flush(i_target_rank, win_list_rank);
  // printf("[%i] run_this_test after : %i \n",i_rank, list_rank_for_test[beg_cur_test]);

  MPI_Win_unlock(i_target_rank, win_list_rank);

}

// --------------------------------------------------------------------------------------
void select_one_test()
{

}

// --------------------------------------------------------------------------------------
void make_decision()
{

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
  // auto dtest_proc = setup_test_distribution(comm, n_rank_for_test);
  std::vector<int> dtest_proc(n_rank+1, n_tot_test);
  dtest_proc[0] = 0;


  // For each test we store a distributed array that contains the number of rank that can execute the test
  // Increment by one via MPI_Put
  std::vector<int> count_rank_for_test(dtest_proc[i_rank+1] - dtest_proc[i_rank], 0);

  // For each test we need the list of rank (in global communicator) that execute the test
  std::vector<int> list_rank_for_test_idx(n_rank_for_test.size()+1, 0);
  for(int i = 0; i < static_cast<int>(n_rank_for_test.size()); ++i) {
    list_rank_for_test_idx[i+1] = list_rank_for_test_idx[i] + n_rank_for_test[i];
  }
  int beg_cur_proc_test = list_rank_for_test_idx[dtest_proc[i_rank  ]]; // 1er  test on the current proc
  int end_cur_proc_test = list_rank_for_test_idx[dtest_proc[i_rank+1]]; // Last test on the current proc
  // std::vector<int> list_rank_for_test(end_cur_proc_test - beg_cur_proc_test, -10);
  std::vector<int> list_rank_for_test(list_rank_for_test_idx.back(), -10);
  // printf("[%i] beg : %i | end : %i | size = %i \n", i_rank, beg_cur_proc_test, end_cur_proc_test, end_cur_proc_test-beg_cur_proc_test);

  MPI_Win win_count_rank_for_test;
  MPI_Win_create(count_rank_for_test.data(),              /* base ptr */
                 count_rank_for_test.size()*sizeof(int),  /* size     */
                 sizeof(int),                             /* disp_unit*/
                 MPI_INFO_NULL,
                 comm,
                 &win_count_rank_for_test);

  // Assez subtile ici car on partage un morceau du tableau mais on le garde global pour stocké nos affaires
  // Si on est sur le rang qui posséde la window il faut assuré la cohérence RMA / Mémoire
  MPI_Win win_list_rank_for_test;
  MPI_Win_create(&list_rank_for_test[beg_cur_proc_test],               /* base ptr */
                 (end_cur_proc_test - beg_cur_proc_test)*sizeof(int),  /* size     */
                 sizeof(int),                                         /* disp_unit*/
                 MPI_INFO_NULL,
                 comm,
                 &win_list_rank_for_test);

  // MPI_Win_post(world_group, 0, win_count_rank_for_test);
  // MPI_Win_start(world_group, 0, win_count_rank_for_test);

  // III/ Setup
  std::vector<int> remote_copy_count_rank(2, -1); // []
  // int i_test_g = 1;

  MPI_Barrier(comm);
  // if(i_rank == 0 ){
  //   auto res = big_loop();
  //   printf("res = %12.5e \n", res);
  // }


  std::vector<int> order(n_rank_for_test.size());
  std::iota(begin(order), end(order), 0);
  int beg = 0;
  int end = n_rank_for_test.size();
  int step = 1;
  // if(i_rank == n_rank-1){
  if(i_rank > n_rank/2-1){
    for(int i = 0; i < end; ++i ){
      order[end-i-1] = i;
      // printf("order[%i]\n", end-i-1);
    }
    beg = n_rank_for_test.size();
    end = 0;
    step = -1;
  }

  // for(int i_test_g = 0; i_test_g < static_cast<int>(n_rank_for_test.size()); ++i_test_g){
  for(int i = 0; i < static_cast<int>(n_rank_for_test.size()); ++i){
    int i_test_g = order[i];
    // win_p0 : [ 0, 0 ] // win_p1 : [ 0 ]
    bool last_decision_failed = false; // 0 = false / 1 = true
    // if(i_rank == 0){
    //   last_decision_failed = true;
    // }
    bool run_this_test        = false;

    // MPI_Win_fence(0, win_count_rank_for_test);
    // MPI_Win_fence(0, win_list_rank_for_test);
    // MPI_Win_fence(0, win_count_rank_for_test);
    // MPI_Win_start(world_group, 0, win_count_rank_for_test);

    if(!last_decision_failed ){

      int i_target_rank = std_e::interval_index(i_test_g, dtest_proc);

      // --------------------------------------------------------------------------------
      if(n_rank_for_test[i_test_g] <=  n_rank){
        run_this_test = update_window_to_match_test(dtest_proc,
                                                    n_rank_for_test,
                                                    list_rank_for_test_idx,
                                                    i_test_g,
                                                    remote_copy_count_rank,
                                                    win_count_rank_for_test,
                                                    win_list_rank_for_test,
                                                    i_rank);
      } else {
        // On passe car sinon on deadlock a coup sure
      }

      // --------------------------------------------------------------------------------
      remote_copy_count_rank[0] = -10;

      // --------------------------------------------------------------------------------
      bool is_ready_with_other = false;
      while( !is_ready_with_other && run_this_test ){

        // printf("[%i] On n'attends pas Patrick - MPI_Win_lock \n", i_rank);
        MPI_Win_lock(MPI_LOCK_SHARED, i_target_rank, 0, win_count_rank_for_test);
        // printf("[%i] On n'attends pas Patrick - MPI_Win_lock end \n", i_rank);

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
        MPI_Win_unlock(i_target_rank, win_count_rank_for_test);
        is_ready_with_other = remote_copy_count_rank[0] == n_rank_for_test[i_test_g];
        // printf("[%i] I'm waiting my bro is_ready_with_other :: %i | run_this_test :: %i | remote_copy_count_rank :: %i \n", i_rank, is_ready_with_other, run_this_test, remote_copy_count_rank[0]);
      }
      // --------------------------------------------------------------------------------

      // MPI_Win_fence(0, win_count_rank_for_test);
      // MPI_Win_fence(0, win_list_rank_for_test);
      // MPI_Win_sync(win_count_rank_for_test);
      // MPI_Win_sync(win_list_rank_for_test);
      // --------------------------------------------------------------------------------

      if(run_this_test) {

        // > Update the rank list localy
        update_list_rank_for_test(dtest_proc,
                                  n_rank_for_test,
                                  list_rank_for_test,
                                  list_rank_for_test_idx,
                                  i_test_g,
                                  i_target_rank,
                                  win_list_rank_for_test,
                                  i_rank);

        // Prepare group
        int beg_cur_test  = list_rank_for_test_idx[i_test_g];
        MPI_Group test_group;
        MPI_Group_incl(world_group,
                       n_rank_for_test[i_test_g],
                       &list_rank_for_test[beg_cur_test],
                       &test_group);

        int i_rank_group;
        int n_rank_group;
        MPI_Group_rank(test_group, &i_rank_group);
        MPI_Group_size(test_group, &n_rank_group);
        MPI_Comm test_comm;
        assert( i_rank_group != MPI_UNDEFINED);
        if(i_rank_group != MPI_UNDEFINED){
          printf("    [%i] Execute test %i \n", i_rank, i_test_g);
          MPI_Comm_create_group(comm, test_group, i_test_g, &test_comm);
        }
        assert(n_rank_group == n_rank_for_test[i_test_g]);

        // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_count_rank_for_test);
        // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_list_rank_for_test);
        // MPI_Win_flush(i_rank, win_count_rank_for_test);
        // MPI_Win_flush(i_rank, win_list_rank_for_test);
        // MPI_Win_unlock(i_rank, win_count_rank_for_test);
        // MPI_Win_unlock(i_rank, win_list_rank_for_test);

        // MPI_Win_complete(win_count_rank_for_test);
        // MPI_Win_wait(win_count_rank_for_test);
        // Listen RDMA :
        // ....
        // std::mutex<int> val = 1;
        // std::atomic_bool cancel = false;
        // auto t1 = std::async(std::launch::async, [&](std::atomic_bool& lcancel){
        //   printf("[%i] begin wait %i \n", i_rank, i_test_g);
        //   while(lcancel == 1){
        //     MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_list_rank_for_test);
        //     MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_count_rank_for_test);
        //     // MPI_Win_flush(i_rank, win_list_rank_for_test);
        //     // MPI_Win_flush(i_rank, win_count_rank_for_test);
        //     MPI_Win_flush_all(win_list_rank_for_test);
        //     MPI_Win_flush_all(win_count_rank_for_test);
        //     MPI_Win_unlock(i_target_rank, win_list_rank_for_test);
        //     MPI_Win_unlock(i_target_rank, win_count_rank_for_test);
        //   }
        //   printf("[%i] end wait for %i \n", i_rank, i_test_g);
        // }, std::ref(cancel));
        tests_suite[i_test_g](test_comm);
        // cancel = true;
        // val = 0;
        // std::chrono::system_clock::time_point zero =  std::chrono::system_clock::now() + std::chrono::seconds(0);
        // t1.wait();

        MPI_Comm_free(&test_comm);
        MPI_Group_free(&test_group);

        // MPI_Win_fence(0, win_count_rank_for_test);
        // MPI_Win_fence(0, win_list_rank_for_test);


      } else {
        // Attention il faut faire i_test++ si et seulement si le test est bien executé
        // i_test_g++;
      }
    }
  }

  // MPI_Win_complete(win_count_rank_for_test);
  // MPI_Win_wait(win_count_rank_for_test);
  // MPI_Barrier(comm);

  // Free
  MPI_Group_free(&world_group);
  MPI_Win_free(&win_list_rank_for_test);
  MPI_Win_free(&win_global_n_rank_available);
  MPI_Win_free(&win_count_rank_for_test);
}
