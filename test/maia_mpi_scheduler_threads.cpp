#include <iostream>
#include <thread>
#include <condition_variable>
#include <algorithm>
#include <vector>
#include "std_e/log.hpp"

#include "std_e/parallel/mpi.hpp"

using namespace std::chrono_literals;

struct t {
  int i;
  auto operator()() const {
    //std::cout << "hello from test " << i << " and rank " << std_e::rank(MPI_COMM_WORLD) << "\n";
    std::this_thread::sleep_for(100ms);
  }
};

const std::vector<t> tests = {{10},{11},{12},{13},{14}};
const std::vector<int> test_nb_procs = {10,8,6,4,3};
//const std::vector<int> test_nb_procs = {1,1,1,1,1};

enum status {
  not_started, run, done
};
std::vector<status> tests_status(tests.size(),not_started);
std::vector<std::vector<int>> procs_by_test(tests.size(),std::vector<int>{});
std::vector<bool> av_proc;
std::mutex mut;
std::condition_variable data_cond;


const int test_to_exec_tag = 42;
const int ranks_test_to_exec_tag = 43;
const int test_done_tag = 44;

void schedule_test(int i_test) {
  int nb_ranks = std_e::nb_ranks(MPI_COMM_WORLD);
  int remaining_procs_to_assign = test_nb_procs[i_test];
  int i=0;
  while (remaining_procs_to_assign) {
    if (av_proc[i]) {
      av_proc[i] = false;
      tests_status[i_test] = run;
      procs_by_test[i_test].push_back(i);
      --remaining_procs_to_assign;
    }
    ++i;
    STD_E_ASSERT(i<=nb_ranks);
  }

  for (int i_rank=0; i_rank<nb_ranks; ++i_rank) {
    int rank_active_in_test = 0;
    if (std::find(begin(procs_by_test[i_test]),end(procs_by_test[i_test]),i_rank)!=end(procs_by_test[i_test])) {
      rank_active_in_test = 1;
    }
    // MPI_Request req0, req1;
    // MPI_Status status;
    MPI_Send(&i_test, 1, MPI_INT, /*to rank*/i_rank, test_to_exec_tag, MPI_COMM_WORLD);
    MPI_Send(&rank_active_in_test, 1, MPI_INT, /*to rank*/i_rank, ranks_test_to_exec_tag, MPI_COMM_WORLD);
  }
}

int test_runnable_with_avail_procs() {
  int nb_tests = tests_status.size();

  int nb_avail_proc = std::count(begin(av_proc),end(av_proc),true);

  for (int i=0; i<nb_tests; ++i) {
    if (tests_status[i]==not_started && test_nb_procs[i]<=nb_avail_proc) {
      return i;
    }
  }
  return -1;
}

void schedule_tests() {
  // int nb_tests = tests.size();
  int nb_remaining_tests = std::count(begin(tests_status),end(tests_status),not_started);
  while (nb_remaining_tests) {
    std::unique_lock lk(mut);
    int i_test = test_runnable_with_avail_procs();
    if (i_test!=-1) { // enought procs avail
      schedule_test(i_test);
    } else {
      data_cond.wait(lk,[]{return test_runnable_with_avail_procs()!=-1;});
    }
    nb_remaining_tests = std::count(begin(tests_status),end(tests_status),not_started);
  }

  // send done message
  int nb_ranks = std_e::nb_ranks(MPI_COMM_WORLD);
  for (int i_rank=0; i_rank<nb_ranks; ++i_rank) {
    int i_test = -1; // end test
    MPI_Send(&i_test, 1, MPI_INT, /*to rank*/i_rank, test_to_exec_tag, MPI_COMM_WORLD);
  }
  ELOG(procs_by_test);
}
void recv_tests() {
  int remain_tests = tests.size();
  while (remain_tests--) {
    int i_test;
    MPI_Status status;
    MPI_Request req;
    MPI_Irecv(&i_test, 1, MPI_INT, MPI_ANY_SOURCE, test_done_tag, MPI_COMM_WORLD,&req);
    MPI_Wait(&req,&status);
    {
      std::lock_guard lk(mut);
      tests_status[i_test] = done;
      for (int proc : procs_by_test[i_test]) {
        av_proc[proc] = true;
      }
    }
    data_cond.notify_one();
  }
}

void exec_test(int i_test, MPI_Comm&& sub_comm) {
  tests[i_test]();
  MPI_Barrier(sub_comm); // TODO2
  if (std_e::rank(sub_comm)==0) { // rank 0 of sub_comm responsible for reporting test ended
    MPI_Send(&i_test, 1, MPI_INT, /*to rank*/0, test_done_tag, MPI_COMM_WORLD);
  }
  MPI_Comm_free(&sub_comm);
}
void launch_tests() {
  std::thread exec_thread;

  // int launched_test = 0;
  while (1) {
    MPI_Status status;
    MPI_Request req;
    int i_test;
    MPI_Irecv(&i_test, 1, MPI_INT, /*from rank*/0, test_to_exec_tag, MPI_COMM_WORLD, &req);
    MPI_Wait(&req,&status);

    if (i_test == -1) { // no test to exec anymore
      break;
    } else {
      int rank_active_in_test;
      MPI_Recv(&rank_active_in_test, 1, MPI_INT, /*from rank*/0, ranks_test_to_exec_tag, MPI_COMM_WORLD, &status);

      // create sub_comm and launch test
      MPI_Comm sub_comm;
      int color = MPI_UNDEFINED;
      if (rank_active_in_test) {
        color = 0;
      }
      int comm_world_rank = std_e::rank(MPI_COMM_WORLD);
      MPI_Comm_split(MPI_COMM_WORLD, color, comm_world_rank, &sub_comm);

      if (sub_comm != MPI_COMM_NULL) {
        if (exec_thread.joinable()) { exec_thread.join(); } // make sure previous test finish (useful?)
        exec_thread = std::thread(exec_test,i_test,std::move(sub_comm));
      }
    }
  }

  if (exec_thread.joinable()) { exec_thread.join(); } // make sure previous test finish (useful?)
}

int main(int argc, char** argv) {
  int provided_thread;
  // MPI_THREAD_SINGLE MPI_THREAD_FUNNELED MPI_THREAD_SERIALIZED MPI_THREAD_MULTIPLE
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread);
  ELOG(provided_thread);
  STD_E_ASSERT(provided_thread==MPI_THREAD_MULTIPLE);

  int nb_ranks = std_e::nb_ranks(MPI_COMM_WORLD);
  av_proc = std::vector<bool>(nb_ranks,true);

  if (std_e::rank(MPI_COMM_WORLD)==0) {

    std::thread t0(schedule_tests);
    std::thread t1(recv_tests);
    launch_tests();
    t0.join();
    t1.join();
  } else {
    launch_tests();
  }

  MPI_Finalize();
  return 0;
}
