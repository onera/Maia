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
// void prepare_buffer(std::vector<int>& buffer,
//                     int               )


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

  // Well we need to setup dynamicly a kind of master
  // We always begin with the first rank to inialize the process !
  int n_rank_available = n_rank;
  std::vector<int> ranks_available(n_rank); // The first is the master
  std::iota(begin(ranks_available), end(ranks_available), 0);

  std::vector<int> buffer(n_rank+3);

  // Setup
  ranks_available[0] = 0;

  // Initialiase the recursion
  int i_test_g = 0;
  int n_info = n_rank_available - n_rank_for_test[i_test_g] + 3;
  buffer[0] = i_test_g+1;
  buffer[1] = 0;
  buffer[2] = n_rank_available;
  for(int i = 0; i < n_rank_available; ++i){
    buffer[i+3] = ranks_available[n_rank_for_test[i_test_g]+i];
  }
  if(i_rank == 0){
    MPI_Send(buffer.data(), n_info, MPI_INT,
             0, i_test_g, comm); // Send to first rank
  }

  // while(i_test_g < n_tot_test) {
  while(i_test_g < 1) {

    // Donc chaque rang se met en attente !
    int flag = 0;
    MPI_Status status;
    while(!flag) {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
      // printf("[%i] Waiting for a message : %i \n ", i_rank, flag);
    }

    // Receive the action :
    // 0 : you're the master let choose a job
    // 1 : Well, we are a slave
    MPI_Status status_recv;
    int n_info;
    MPI_Get_count(&status, MPI_INT, &n_info);

    std::vector<int> incoming_info(n_info);
    MPI_Recv(incoming_info.data(), n_info, MPI_INT, status.MPI_SOURCE,
             status.MPI_TAG, comm, &status_recv);

    /* Find out another test and recurse */
    i_test_g = incoming_info[0];
    int n_rank_available = incoming_info[2];
    for(int i = 0; i < n_rank_available; ++i) {
      ranks_available[i] = incoming_info[i+2];
    }

    if(incoming_info[1] == 0){
      printf("[%i] is the master now ! \n", i_rank);

      // Send to other proc correct information to be launch
      // i_test_g = choose_on_test()
      buffer[0] = i_test_g;
      buffer[1] = 0;
      buffer[2] = n_rank_for_test[i_test_g];
      for(int i = 0; i < n_rank_for_test[i_test_g]; ++i){
        buffer[i+3] = ranks_available[n_rank_for_test[i_test_g]+i];
      }
      for(int idx = 1; idx < n_rank_for_test[i_test_g]; ++idx){ // On exclue le premier car c'est le master
        MPI_Send(buffer.data(), n_info, MPI_INT,
                 ranks_available[idx], i_test_g, comm); // Send to first rank
      }

      // On envoie également au proc d'aprés l'information pour le nouveu job


    } else if ( incoming_info[1] == 1) {
      // Slave
      printf("[%i] is the slave now ! \n", i_rank);

    }

    // Create the subgroup
    MPI_Group test_group;
    MPI_Group_incl(world_group,
                   n_rank_for_test[i_test_g],
                   &buffer[3],
                   &test_group);


    i_test_g++;
  }


  // If the current is the master we do multiple things :
  //   - This rank have the responsbility to choose the optimal test to launch
  //   - He send information to the available rank - And setup the group
  // if(ranks_available[0] == i_rank){
  //   // launch job
  //   // tests_suite[i_test_g](test_comm);
  // } else {
  //   // Wait for a message in order to begin a test
  //   int flag = 0;
  //   MPI_Status status;
  //   while(!flag) {
  //     MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
  //     printf("[%i] Waiting for a message : %i \n ", i_rank, flag);
  //   }

  //   MPI_Status status_recv;
  //   int n_info;
  //   MPI_Get_count(&status, MPI_INT, &n_info);

  //   std::vector<int> incoming_info(n_info);
  //   MPI_Recv(incoming_info.data(), n_info, MPI_INT, status.MPI_SOURCE,
  //            status.MPI_TAG, comm, &status_recv);

  //   /* Find out another test and recurse */
  //   i_test_g = incoming_info[0];
  //   int n_rank_available = incoming_info[1];
  //   for(int i = 0; i < n_rank_available; ++i) {
  //     ranks_available[i] = incoming_info[i+2];
  //   }
  // }

  MPI_Group_free(&world_group);
}
