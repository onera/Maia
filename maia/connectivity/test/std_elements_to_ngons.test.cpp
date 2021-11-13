#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

#include "maia/connectivity/std_elements_to_ngons.hpp"
#include "std_e/parallel/mpi/base.hpp"
#include "std_e/log.hpp" // TODO
#include "std_e/multi_array/utils.hpp" // TODO


using namespace cgns;

PYBIND_MPI_TEST_CASE("std_elements_to_ngons",2) {
  int rk = std_e::rank(test_comm);
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_"+std::to_string(rk)+".yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  maia::std_elements_to_ngons(z,test_comm);
  //ELOG(z);

  auto elt_type = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons")[0];
  auto range = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons/ElementRange");
  auto connec = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons/ElementConnectivity");
  auto parents = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGons/ParentElements");
  ELOG(elt_type);
  ELOG(range);
  ELOG(connec);
  ELOG(parents.extent());
  ELOG(parents);
  CHECK( elt_type == (I4)cgns::NGON_n);
  CHECK( range == std::vector<I4>{1,18} );
  //MPI_CHECK(0, connec == std::vector<I4>{} ); // TODO
  //MPI_CHECK(1, connec == std::vector<I4>{} ); // TODO
  MPI_CHECK(0, parents.extent() == std_e::multi_index<I8,2>{8,2} );
  MPI_CHECK(1, parents.extent() == std_e::multi_index<I8,2>{10,2} );
  //MPI_CHECK(0, parents == cgns::md_array<I4,2>{{17,0}} ); // TODO
  //MPI_CHECK(1, parents == cgns::md_array<I4,2>{{18,0}} ); // TODO
}
