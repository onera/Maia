#include "mpi.h"
#include "doctest/extensions/doctest_mpi.h"
#include "maia/algo/part/cgns_registry/cgns_registry.hpp"



// ---------------------------------------------------------------------------------
constexpr auto cfg1_path_base     = "/cgns_base";
constexpr auto cfg1_path_zone1    = "/cgns_base/Zone1";
constexpr auto cfg1_path_zone2    = "/cgns_base/Zone2";
constexpr auto cfg1_path_zone1_j1 = "/cgns_base/Zone1/ZoneGridConnectivity/Join1";
constexpr auto cfg1_path_zone2_j2 = "/cgns_base/Zone2/ZoneGridConnectivity/Join2";

// ---------------------------------------------------------------------------------
inline
cgns_paths_by_label generate_case_1_1_proc(){
  cgns_paths_by_label cgns_paths;

  add_path(cgns_paths, cfg1_path_base, CGNS::Label::CGNSBase_t);

  add_path(cgns_paths, cfg1_path_zone1, CGNS::Label::Zone_t);
  add_path(cgns_paths, cfg1_path_zone2, CGNS::Label::Zone_t);

  add_path(cgns_paths, cfg1_path_zone1_j1, CGNS::Label::GridConnectivity1to1_t);
  add_path(cgns_paths, cfg1_path_zone2_j2, CGNS::Label::GridConnectivity1to1_t);

  return cgns_paths;
}

// ---------------------------------------------------------------------------------
inline
cgns_paths_by_label generate_case_1_2_proc(MPI_Comm& comm){
  cgns_paths_by_label cgns_paths;

  int n_rank_l, i_rank_l;
  MPI_Comm_size(comm, &n_rank_l);
  MPI_Comm_rank(comm, &i_rank_l);
  assert(n_rank_l == 2);

  if( i_rank_l == 0) {
    add_path(cgns_paths, cfg1_path_base    , CGNS::Label::CGNSBase_t);
    add_path(cgns_paths, cfg1_path_zone1   , CGNS::Label::Zone_t);
    add_path(cgns_paths, cfg1_path_zone1_j1, CGNS::Label::GridConnectivity1to1_t);
  } else {
    add_path(cgns_paths, cfg1_path_base    , CGNS::Label::CGNSBase_t);
    add_path(cgns_paths, cfg1_path_zone2   , CGNS::Label::Zone_t);
    add_path(cgns_paths, cfg1_path_zone2_j2, CGNS::Label::GridConnectivity1to1_t);
  }

  return cgns_paths;
}

// ---------------------------------------------------------------------------------
MPI_TEST_CASE("[cgns_registry] Generate global numbering [1 proc]",1) {
  cgns_paths_by_label conf_1 = generate_case_1_1_proc();

  cgns_registry cgns_reg1 = cgns_registry(conf_1, test_comm);

  /* CGNS Path have the same id around all procs */
  int g_id_base  = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_base    , CGNS::Label::CGNSBase_t            );
  int g_id_zone1 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone1   , CGNS::Label::Zone_t                );
  int g_id_zone2 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone2   , CGNS::Label::Zone_t                );
  int g_id_join1 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone1_j1, CGNS::Label::GridConnectivity1to1_t);
  int g_id_join2 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone2_j2, CGNS::Label::GridConnectivity1to1_t);

  CHECK(g_id_base  == 1);
  CHECK(g_id_zone1 == 1);
  CHECK(g_id_zone2 == 2);
  CHECK(g_id_join1 == 1);
  CHECK(g_id_join2 == 2);

}


// ---------------------------------------------------------------------------------
MPI_TEST_CASE("[cgns_registry] Generate global numbering [2 proc]",2) {
  cgns_paths_by_label conf_1 = generate_case_1_2_proc(test_comm);

  cgns_registry cgns_reg1 = cgns_registry(conf_1, test_comm);

  int n_rank_l, i_rank_l;
  MPI_Comm_size(test_comm, &n_rank_l);
  MPI_Comm_rank(test_comm, &i_rank_l);

  int g_id_base  = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_base    , CGNS::Label::CGNSBase_t);

  DOCTEST_MPI_CHECK(0, g_id_base == 1);
  DOCTEST_MPI_CHECK(1, g_id_base == 1);

  std::string b_path  = get_path_from_global_id_and_type(cgns_reg1, g_id_base, CGNS::Label::CGNSBase_t);

  DOCTEST_MPI_CHECK(0, b_path == cfg1_path_base);
  DOCTEST_MPI_CHECK(1, b_path == cfg1_path_base);


  if(i_rank_l == 0){

    int g_id_zone1 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone1   , CGNS::Label::Zone_t                );
    int g_id_join1 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone1_j1, CGNS::Label::GridConnectivity1to1_t);

    DOCTEST_MPI_CHECK(0, g_id_zone1 == 1);
    DOCTEST_MPI_CHECK(0, g_id_join1 == 1);

    std::string z1_path = get_path_from_global_id_and_type(cgns_reg1, g_id_zone1, CGNS::Label::Zone_t);
    std::string j1_path = get_path_from_global_id_and_type(cgns_reg1, g_id_join1, CGNS::Label::GridConnectivity1to1_t);

    DOCTEST_MPI_CHECK(0, z1_path == cfg1_path_zone1   );
    DOCTEST_MPI_CHECK(0, j1_path == cfg1_path_zone1_j1);

  } else {

    int g_id_zone2 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone2   , CGNS::Label::Zone_t                );
    int g_id_join2 = get_global_id_from_path_and_type(cgns_reg1, cfg1_path_zone2_j2, CGNS::Label::GridConnectivity1to1_t);

    DOCTEST_MPI_CHECK(1, g_id_zone2 == 2);
    DOCTEST_MPI_CHECK(1, g_id_join2 == 2);

    std::string z2_path = get_path_from_global_id_and_type(cgns_reg1, g_id_zone2, CGNS::Label::Zone_t);
    std::string j2_path = get_path_from_global_id_and_type(cgns_reg1, g_id_join2, CGNS::Label::GridConnectivity1to1_t);

    DOCTEST_MPI_CHECK(1, z2_path == cfg1_path_zone2   );
    DOCTEST_MPI_CHECK(1, j2_path == cfg1_path_zone2_j2);

  }

}
