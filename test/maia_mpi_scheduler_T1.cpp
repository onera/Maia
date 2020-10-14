#include <mpi.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <functional>
#include <unistd.h>
#include "maia/utils/mpi_scheduler.hpp"

// -------------------------------------------------------------------------
void init_log(){
  // A faire
}

// -------------------------------------------------------------------------
void banner_test(MPI_Comm& comm, std::string comment){
  int n_rank, i_rank;
  MPI_Comm_size(comm, &n_rank);
  MPI_Comm_rank(comm, &i_rank);
  std::cout << "Execute test : " << comment << " on [" << i_rank << "/" << n_rank << "]" << std::endl;
}

// -------------------------------------------------------------------------
void test_1(MPI_Comm& comm)
{
  banner_test(comm, "test_1");
  printf("Test 1 beg \n");
  sleep(5);
  printf("Test 1 end \n");
}

// -------------------------------------------------------------------------
void test_2(MPI_Comm& comm)
{
  banner_test(comm, "test_1");
  printf("Test 2 beg \n");
  sleep(5);
  printf("Test 2 end \n");
}

// -------------------------------------------------------------------------
void setup_test(MPI_Comm&                                    comm,
                std::vector<int>&                            n_rank_for_test,
                std::vector<std::function<void(MPI_Comm&)>>& tests_suite)
{
  test_1(comm);

  tests_suite[0](comm);
  tests_suite[1](comm);

  // MPI_Win_create()

  // MPI_Win_lock()
  // MPI_Put()
  // MPI_Win_flush() // Necessary if lock ?
  // MPI_Win_unlock()


};

// -------------------------------------------------------------------------
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  // MPI_Init(NULL, NULL);
  int n_rank, i_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);

  MPI_Comm g_comm = MPI_COMM_WORLD;



  std::vector<int> n_rank_for_test = {1, 2};
  std::vector<std::function<void(MPI_Comm&)>> tests_suite = {&test_1, &test_2};
  // setup_test(g_comm, n_rank_for_test, tests_suite);
  run_scheduler(g_comm, n_rank_for_test, tests_suite);
  return 0;

  int dn_test = 1; // Chaque rang en posséde 1
  int* test_status;
  MPI_Win win;
  int err_alloc = MPI_Win_allocate(dn_test * sizeof(int),
                                   sizeof(int),
                                   MPI_INFO_NULL,
                                   g_comm,
                                   &test_status,
                                   &win);

  MPI_Group world_group;
  MPI_Comm_group(g_comm, &world_group);

  test_status[0] = -i_rank*10;
  MPI_Barrier(g_comm);

  // MPI_Win_start(world_group, 0, win);
  // MPI_Win_post(world_group, 0, win);

  assert(err_alloc == MPI_SUCCESS);
  if(i_rank == 0){
    sleep(2);
    int target = 1;
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
    printf("rank [%i] lock window \n", i_rank);
    int err_put = MPI_Put(&i_rank,    /* origin_addr     */
                          1,          /* origin_count    */
                          MPI_INT,    /* origin_datatype */
                          target,     /* target_rank     */
                          0,          /* target_disp     */
                          1,          /* target_count    */
                          MPI_INT,    /* target_datatype */
                          win);       /* win             */
    assert(err_put == MPI_SUCCESS);

    MPI_Win_flush(target, win);
    MPI_Win_unlock(target, win);
    printf("rank [%i] finish \n", i_rank);
  }

  // MPI_Win_wait(win);

  if(i_rank == 1){
    // sleep(2);
    int target = 1;
    while(test_status[0] != 0){
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win);
      printf("rank [%i] lock window with data = [%i] \n", i_rank, test_status[0]);
      MPI_Win_unlock(target, win);
    }
  }

  MPI_Barrier(g_comm);
  // MPI_Win_flush_all(win);
  int err_free = MPI_Win_free(&win);
  assert(err_free == MPI_SUCCESS);

  // MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm,
  //  1251                    MPI_Win *win) MPICH_API_PUBLIC;
  // MPI_Win_create()

  MPI_Finalize();
}
