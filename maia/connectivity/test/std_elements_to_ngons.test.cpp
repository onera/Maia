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

  auto elt_type = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons")[0];
  auto range = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons/ElementRange");
  auto eso = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons/ElementStartOffset");
  auto connec = cgns::get_node_value_by_matching<I4>(b,"Zone/NGons/ElementConnectivity");
  auto parents = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGons/ParentElements");
  CHECK( elt_type == (I4)cgns::NGON_n);
  CHECK( range == std::vector<I4>{1,18} );
  MPI_CHECK(0, eso == std::vector<I4>{0,4,8,12,16,20,24,28,32} );
  MPI_CHECK(0, connec == std::vector<I4>{1,6,9,4,6,11,14,9,3,5,10,8,8,10,15,13,1,2,7,6,2,3,8,7,6,7,12,11,7,8,13,12} );
  MPI_CHECK(1, eso == std::vector<I4>{32,36,40,44,48,51,54,57,61,65,69} );
  MPI_CHECK(1, connec == std::vector<I4>{4,9,10,5,9,14,15,10,1,4,5,2,11,12,15,14,2,5,3,12,13,15,7,8,10,2,7,10,5,6,7,10,9,7,12,15,10} );
  MPI_CHECK(0, parents.extent() == std_e::multi_index<I8,2>{8,2} );
  MPI_CHECK(1, parents.extent() == std_e::multi_index<I8,2>{10,2} );
  MPI_CHECK(0, parents == cgns::md_array<I4,2>{{1,0},{2,0},{3,0},{4,0},{1,0},{3,0},{2,0},{4,0}} );
  MPI_CHECK(1, parents == cgns::md_array<I4,2>{{1,0},{2,0},{1,0},{2,0},{3,0},{4,0},{3,4},{3,1},{1,2},{4,2}} );
}
