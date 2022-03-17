#include "std_e/unit_test/doctest_pybind.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"

#include "maia/connectivity/std_elements_to_ngons.hpp"
#include "maia/transform/poly_algorithm.hpp"
#include "std_e/log.hpp"

using cgns::I4;
using cgns::I8;

PYBIND_TEST_CASE("ngon_new_to_old") {
  // setup
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_01.yaml");
  cgns::tree b = maia::to_node(yaml_tree);
  cgns::tree& z = cgns::get_node_by_name(b,"Zone");
  maia::std_elements_to_ngons(z,MPI_COMM_SELF); // Note: we are not testing that, its just a way to get an ngon test

  // apply tested function
  maia::ngon_new_to_old(z);

  // check
  auto connec = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementConnectivity");
  CHECK(connec == std::vector{
      4, 1,6, 9,4,  4, 6 ,11,14, 9,  4, 3, 5,10, 8,  4,  8,10,15,13,
      4, 1,2, 7,6,  4, 2 , 3, 8, 7,  4, 6, 7,12,11,  4,  7, 8,13,12,
      4, 4,9,10,5,  4, 9 ,14,15,10,  4, 1, 4, 5, 2,  4, 11,12,15,14,
      3, 2,5,3   ,  3, 12,13,15   ,  3, 7, 8,10,
      4, 2,7,10,5,  4, 6 , 7,10, 9,  4, 7,12,15,10} );

  auto pe = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGON_n/ParentElements");
  CHECK( pe == cgns::md_array<I4,2>{{1,0},{2,0},{3,0},{4,0},{1,0},{3,0},{2,0},{4,0},{1,0},
                                    {2,0},{1,0},{2,0},{3,0},{4,0},{3,4},{3,1},{1,2},{4,2}} );

  CHECK( !cgns::has_node(b,"Zone/NGON_n/ElementStartOffset") );
}

PYBIND_TEST_CASE("ngon_old_to_new") {
  // setup
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_01.yaml");
  cgns::tree b = maia::to_node(yaml_tree);
  cgns::tree& z = cgns::get_node_by_name(b,"Zone");
  maia::std_elements_to_ngons(z,MPI_COMM_SELF); // We are not testing that, its just a way to get an ngon test
  maia::ngon_new_to_old(z); // Not testing that either

  // apply tested function
  maia::ngon_old_to_new(z);

  // check
  auto eso = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementStartOffset");
  CHECK( eso == std::vector{0,4,8,12,16,20,24,28,32,36,40,44,48,51,54,57,61,65,69} );

  auto connec = cgns::get_node_value_by_matching<I4>(b,"Zone/NGON_n/ElementConnectivity");
  CHECK( connec == std::vector{1,6, 9,4,   6,11,14, 9,   3, 5,10, 8,   8,10,15,13,
                               1,2, 7,6,   2, 3, 8, 7,   6, 7,12,11,   7, 8,13,12,
                               4,9,10,5,   9,14,15,10,   1, 4, 5, 2,  11,12,15,14,
                               2,5, 3,     12,13,15  ,   7, 8,10,
                               2,7,10,5,   6, 7,10, 9,   7,12,15,10} );

  auto pe = cgns::get_node_value_by_matching<I4,2>(b,"Zone/NGON_n/ParentElements");
  CHECK( pe == cgns::md_array<I4,2>{{19,0},{20,0},{21,0},{22,0},{19,0},{21,0},{20, 0},{22, 0},{19,0},{20,0},{19,0},{20,0},{21,0},{22,0},{21,22},{21,19},{19,20},{22,20}} );
}
