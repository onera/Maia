#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind.hpp"

#include "maia/io/file.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "cpp_cgns/tree_manip.hpp"

#include "maia/algo/dist/elements_to_ngons/elements_to_ngons.hpp"
#include "maia/algo/common/poly_algorithm.hpp"


using cgns::I4;
using cgns::I8;
using cgns::tree;


PYBIND_TEST_CASE("indexed_to_interleaved_connectivity") {
  // setup
  std::string file_name = maia::mesh_dir+"hex_2_prism_2.yaml";
  tree t = maia::file_to_dist_tree(file_name,MPI_COMM_SELF);
  tree& z = cgns::get_node_by_matching(t,"Base/Zone");
  maia::elements_to_ngons(z,MPI_COMM_SELF); // Note: we are not testing that, its just a way to get an ngon test
  tree& ngons = cgns::get_child_by_name(z,"NGON_n");

  // apply tested function
  maia::indexed_to_interleaved_connectivity(ngons);

  // check
  auto connec = cgns::get_node_value_by_matching<I4>(z,"NGON_n/ElementConnectivity");
  CHECK(connec == std::vector{
      4, 1,6, 9,4,    4, 6 ,11,14, 9,    4, 3, 5,10, 8,    4,  8,10,15,13,
      4, 1,2, 7,6,    4, 2 , 3, 8, 7,    4, 6, 7,12,11,    4,  7, 8,13,12,
      4, 4,9,10,5,    4, 9 ,14,15,10,    4, 1, 4, 5, 2,    4, 11,12,15,14,
      3, 2,5,3   ,    3, 12,13,15   ,    3, 7, 8,10,
      4, 2,5,10,7,    4, 6 , 7,10, 9,    4, 7,10,15,12} );

  CHECK( !cgns::has_node(z,"NGON_n/ElementStartOffset") );
}

PYBIND_TEST_CASE("interleaved_to_indexed_connectivity") {
  // setup
  std::string file_name = maia::mesh_dir+"hex_2_prism_2.yaml";
  tree t = maia::file_to_dist_tree(file_name,MPI_COMM_SELF);
  tree& z = cgns::get_node_by_matching(t,"Base/Zone");
  maia::elements_to_ngons(z,MPI_COMM_SELF); // We are not testing that, its just a way to get an ngon test
  tree& ngons = cgns::get_child_by_name(z,"NGON_n");
  maia::indexed_to_interleaved_connectivity(ngons); // Not testing that either

  // apply tested function
  maia::interleaved_to_indexed_connectivity(ngons);

  // check
  auto eso = cgns::get_node_value_by_matching<I4>(z,"NGON_n/ElementStartOffset");
  CHECK( eso == std::vector{0,4,8,12,16,20,24,28,32,36,40,44,48,51,54,57,61,65,69} );

  auto connec = cgns::get_node_value_by_matching<I4>(z,"NGON_n/ElementConnectivity");
  CHECK( connec == std::vector{1,6, 9,4,   6,11,14, 9,   3, 5,10, 8,   8,10,15,13,
                               1,2, 7,6,   2, 3, 8, 7,   6, 7,12,11,   7, 8,13,12,
                               4,9,10,5,   9,14,15,10,   1, 4, 5, 2,  11,12,15,14,
                               2,5, 3,     12,13,15  ,   7, 8,10,
                               2,5,10,7,   6, 7,10, 9,   7,10,15,12} );
}
#endif // C++>17
