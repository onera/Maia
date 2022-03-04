#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

#include "maia/connectivity/std_elements_to_ngons.hpp"
#include "std_e/parallel/mpi/base.hpp"


using namespace cgns;

PYBIND_MPI_TEST_CASE("std_elements_to_ngons",2) {
  int rk = std_e::rank(test_comm);
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_"+std::to_string(rk)+".yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");

  maia::std_elements_to_ngons(z,test_comm);

  auto elt_type = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n")[0];
  auto range = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementRange");
  auto eso = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementStartOffset");
  auto connec = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementConnectivity");
  auto pe = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGON_n/ParentElements");
  auto pp = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGON_n/ParentElementsPosition");
  CHECK( elt_type == (I4)cgns::NGON_n);
  CHECK( range == std::vector<I4>{1,18} );
  MPI_CHECK(0, eso == std::vector<I4>{0,4,8,12,16,20,24,28,32} );
  MPI_CHECK(0, connec == std::vector<I4>{1,6,9,4,6,11,14,9,3,5,10,8,8,10,15,13,1,2,7,6,2,3,8,7,6,7,12,11,7,8,13,12} );
  MPI_CHECK(1, eso == std::vector<I4>{32,36,40,44,48,51,54,57,61,65,69} );
  MPI_CHECK(1, connec == std::vector<I4>{4,9,10,5,9,14,15,10,1,4,5,2,11,12,15,14,2,5,3,12,13,15,7,8,10,2,7,10,5,6,7,10,9,7,12,15,10} );
  MPI_CHECK(0, pe.extent() == std_e::multi_index<I8,2>{8,2} );
  MPI_CHECK(1, pe.extent() == std_e::multi_index<I8,2>{10,2} );
  MPI_CHECK(0, pe == cgns::md_array<I4,2>{{19,0},{20,0},{21,0},{22,0},{19,0},{21,0},{20, 0},{22, 0}} );
  MPI_CHECK(1, pe == cgns::md_array<I4,2>{{19,0},{20,0},{19,0},{20,0},{21,0},{22,0},{21,22},{21,19},{19,20},{22,20}} );
  MPI_CHECK(0, pp == cgns::md_array<I4,2>{{5,0},{5,0},{2,0},{2,0},{2,0},{1,0},{2,0},{1,0}} );
  MPI_CHECK(1, pp == cgns::md_array<I4,2>{{4,0},{4,0},{1,0},{6,0},{4,0},{5,0},{5,4},{3,3},{6,1},{3,3}} );
}
