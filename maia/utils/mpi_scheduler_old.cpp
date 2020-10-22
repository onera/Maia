#include <cassert>
#include <unistd.h>
#include <future>
#include "std_e/algorithm/distribution.hpp"
#include "maia/utils/mpi_scheduler.hpp"
#include "std_e/logging/log.hpp"
#include "std_e/interval/knot_sequence.hpp"

// --------------------------------------------------------------------------------------
void run_scheduler_old(MPI_Comm&                                    comm,
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

  // II/ Setup the distributed rank value for each test
  // auto dtest_proc = setup_test_distribution(comm, n_rank_for_test);
  // Mode chelou to check RDMA access
  std::vector<int> dtest_proc(n_rank+1, n_tot_test);
  dtest_proc[0] = 0;

  // For each test we store a distributed array that contains the number of rank that can execute the test
  // Increment by one via MPI_Put
  // std::vector<int> count_rank_for_test(dtest_proc[i_rank+1] - dtest_proc[i_rank], 0);
  int dn_test = dtest_proc[i_rank+1] - dtest_proc[i_rank];
  int* count_rank_for_test = NULL;
  // printf(" dn_test :: %i \n", dn_test);
  // MPI_Alloc_mem(dn_test * sizeof(int), MPI_INFO_NULL, &count_rank_for_test);

  // For each test we need the list of rank (in global communicator) that execute the test
  std::vector<int> list_rank_for_test_idx(n_rank_for_test.size()+1, 0);
  for(int i = 0; i < static_cast<int>(n_rank_for_test.size()); ++i) {
    list_rank_for_test_idx[i+1] = list_rank_for_test_idx[i] + n_rank_for_test[i];
  }
  int beg_cur_proc_test = list_rank_for_test_idx[dtest_proc[i_rank  ]]; // 1er  test on the current proc
  int end_cur_proc_test = list_rank_for_test_idx[dtest_proc[i_rank+1]]; // Last test on the current proc
  // std::vector<int> list_rank_for_test(end_cur_proc_test - beg_cur_proc_test, -10);

  // std::vector<int> list_rank_for_test(list_rank_for_test_idx.back(), -10);
  int* list_rank_for_test = NULL;
  // MPI_Alloc_mem(list_rank_for_test_idx.back() * sizeof(int), MPI_INFO_NULL, &list_rank_for_test);
  // printf("[%i] beg : %i | end : %i | size = %i \n", i_rank, beg_cur_proc_test, end_cur_proc_test, end_cur_proc_test-beg_cur_proc_test);

  // for(int i = 0; i < list_rank_for_test_idx.back(); ++i){
  //   list_rank_for_test[i] = 0;
  // }

  //
  MPI_Win win_count_rank_for_test;
  MPI_Win win_list_rank_for_test;

  // Solution 1 : vector --> Fail
  // MPI_Win_create(count_rank_for_test.data(),              /* base ptr */
  //                count_rank_for_test.size()*sizeof(int),  /* size     */
  //                sizeof(int),                             /* disp_unit*/
  //                MPI_INFO_NULL,
  //                comm,
  //                &win_count_rank_for_test);

  // Assez subtile ici car on partage un morceau du tableau mais on le garde global pour stocké nos affaires
  // Si on est sur le rang qui posséde la window il faut assuré la cohérence RMA / Mémoire
  // MPI_Win_create(&list_rank_for_test[beg_cur_proc_test],               /* base ptr */
  //                (end_cur_proc_test - beg_cur_proc_test)*sizeof(int),  /* size     */
  //                sizeof(int),                                          /* disp_unit*/
  //                MPI_INFO_NULL,
  //                comm,
  //                &win_list_rank_for_test);

  // Solution 2 : Allocate memory by mpi and attach to window
  // for(int i = 0; i < dn_test; ++i){
  //   count_rank_for_test[i] = 0;
  // }
  // MPI_Win_create(count_rank_for_test,              /* base ptr */
  //                dn_test*sizeof(int),              /* size     */
  //                sizeof(int),                      /* disp_unit*/
  //                MPI_INFO_NULL,
  //                comm,
  //                &win_count_rank_for_test);

  // Assez subtile ici car on partage un morceau du tableau mais on le garde global pour stocké nos affaires
  // Si on est sur le rang qui posséde la window il faut assuré la cohérence RMA / Mémoire
  // MPI_Win_create(&list_rank_for_test[beg_cur_proc_test],               /* base ptr */
  //                (end_cur_proc_test - beg_cur_proc_test)*sizeof(int),  /* size     */
  //                sizeof(int),                                          /* disp_unit*/
  //                MPI_INFO_NULL,
  //                comm,
  //                &win_list_rank_for_test);


  int err_alloc = MPI_Win_allocate(dn_test * sizeof(int),
                                   sizeof(int),
                                   MPI_INFO_NULL,
                                   comm,
                                   &count_rank_for_test,
                                   &win_count_rank_for_test);
  assert(err_alloc == MPI_SUCCESS);

  // // int dn_test_list = end_cur_proc_test - beg_cur_proc_test;
  // int dn_test_list = n_rank_for_test.size()+1;
  int dn_test_list = list_rank_for_test_idx.back();
  MPI_Win_allocate(dn_test_list * sizeof(int),
                   sizeof(int),
                   MPI_INFO_NULL,
                   comm,
                   &list_rank_for_test,
                   &win_list_rank_for_test);


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
  // int beg = 0;
  // int end = n_rank_for_test.size();
  // int step = 1;
  // // if(i_rank == n_rank-1){
  // if(i_rank > n_rank/2-1){
  //   for(int i = 0; i < end; ++i ){
  //     order[end-i-1] = i;
  //     // printf("order[%i]\n", end-i-1);
  //   }
  //   beg = n_rank_for_test.size();
  //   end = 0;
  //   step = -1;
  // }

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
      // remote_copy_count_rank[0] = -10;

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

      // MPI_Win_fence(0, win_list_rank_for_test);
      // MPI_Win_fence(0, win_count_rank_for_test);

      // MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE | MPI_MODE_NOSUCCEED, win_list_rank_for_test);
      // MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE | MPI_MODE_NOSUCCEED, win_count_rank_for_test);
      // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, MPI_MODE_NOCHECK, win_list_rank_for_test);
      // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, MPI_MODE_NOCHECK, win_count_rank_for_test);

      // MPI_Win_sync(win_list_rank_for_test);
      // MPI_Win_sync(win_count_rank_for_test);

      if(run_this_test) {

        printf("[%i] run_this_test : %i \n", i_rank, i_test_g);
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
        // MPI_Group test_group;
        MPI_Group test_group; // = list_group[i_test_g];
        MPI_Group_incl(world_group,
                       n_rank_for_test[i_test_g],
                       &list_rank_for_test[beg_cur_test],
                       &test_group);

        int i_rank_group;
        int n_rank_group;
        MPI_Group_rank(test_group, &i_rank_group);
        MPI_Group_size(test_group, &n_rank_group);
        MPI_Comm test_comm; // = list_comm[i_test_g];
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
        // cancel = true;
        // val = 0;
        // std::chrono::system_clock::time_point zero =  std::chrono::system_clock::now() + std::chrono::seconds(0);
        // t1.wait();


        // MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE | MPI_MODE_NOSUCCEED, win_list_rank_for_test);
        // MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE | MPI_MODE_NOSUCCEED, win_count_rank_for_test);

        // MPI_Win_unlock_all(win_list_rank_for_test);
        // MPI_Win_unlock_all(win_count_rank_for_test);

        // MPI_Win_unlock(i_rank, win_count_rank_for_test);
        // MPI_Win_unlock(i_rank, win_list_rank_for_test);
        tests_suite[i_test_g](test_comm);
        // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_list_rank_for_test);
        // MPI_Win_lock(MPI_LOCK_SHARED, i_rank, 0, win_count_rank_for_test);
        // MPI_Win_flush_all(win_list_rank_for_test);
        // MPI_Win_flush_all(win_count_rank_for_test);
        // MPI_Win_unlock(i_rank, win_list_rank_for_test);
        // MPI_Win_unlock(i_rank, win_count_rank_for_test);

        MPI_Comm_free(&test_comm);
        MPI_Group_free(&test_group);

        // MPI_Win_fence(0, win_count_rank_for_test);
        // MPI_Win_fence(0, win_list_rank_for_test);


      } else {
        // Attention il faut faire i_test++ si et seulement si le test est bien executé
        // i_test_g++;
      }
    }

    // MPI_Win_unlock(i_rank, win_list_rank_for_test);
    // MPI_Win_unlock(i_rank, win_count_rank_for_test);
    // MPI_Win_fence(0, win_list_rank_for_test);
    // MPI_Win_fence(0, win_count_rank_for_test);
  }

  // Free
  MPI_Group_free(&world_group);
  MPI_Win_free(&win_list_rank_for_test);
  MPI_Win_free(&win_count_rank_for_test);
}
