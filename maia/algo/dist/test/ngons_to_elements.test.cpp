#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind.hpp"

#include "maia/io/file.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/algo/dist/elements_to_ngons/elements_to_ngons.hpp"

#include "maia/__old/transform/convert_to_std_elements.hpp"


using cgns::I4;
using cgns::I8;
using cgns::tree;


PYBIND_TEST_CASE("convert_zone_to_std_elements") {
  // setup
  std::string file_name = maia::mesh_dir+"hex_2_prism_2.yaml";
  tree t = maia::file_to_dist_tree(file_name,MPI_COMM_SELF);
  tree& z = cgns::get_node_by_matching(t,"Base/Zone");
  maia::elements_to_ngons(z,MPI_COMM_SELF); // Note: we are not testing that, its just a way to get an ngon test

  // apply tested function
  maia::convert_zone_to_std_elements(z);

  // check
  auto tri_elt_type = cgns::get_node_value_by_matching<I4>(z,"TRI_3")[0];
  auto tri_range    = cgns::get_node_value_by_matching<I4>(z,"TRI_3/ElementRange");
  auto tri_connec   = cgns::get_node_value_by_matching<I4>(z,"TRI_3/ElementConnectivity");

  auto quad_elt_type = cgns::get_node_value_by_matching<I4>(z,"QUAD_4")[0];
  auto quad_range    = cgns::get_node_value_by_matching<I4>(z,"QUAD_4/ElementRange");
  auto quad_connec   = cgns::get_node_value_by_matching<I4>(z,"QUAD_4/ElementConnectivity");

  auto penta_elt_type = cgns::get_node_value_by_matching<I4>(z,"PENTA_6")[0];
  auto penta_range    = cgns::get_node_value_by_matching<I4>(z,"PENTA_6/ElementRange");
  auto penta_connec   = cgns::get_node_value_by_matching<I4>(z,"PENTA_6/ElementConnectivity");

  auto hexa_elt_type = cgns::get_node_value_by_matching<I4>(z,"HEXA_8")[0];
  auto hexa_range    = cgns::get_node_value_by_matching<I4>(z,"HEXA_8/ElementRange");
  auto hexa_connec   = cgns::get_node_value_by_matching<I4>(z,"HEXA_8/ElementConnectivity");

  CHECK( tri_elt_type == cgns::TRI_3 );
  CHECK( tri_range    == std::vector<I4>{1,2} );
  CHECK( tri_connec   == std::vector<I4>{2,5,3,12,13,15} );

  CHECK( quad_elt_type == cgns::QUAD_4 );
  CHECK( quad_range    == std::vector<I4>{3,14} );
  CHECK( quad_connec   == std::vector<I4>{1,6, 9,4,  6,11,14, 9,  3,5,10, 8,   8,10,15,13,
                                          1,2, 7,6,  2, 3, 8, 7,  6,7,12,11,   7, 8,13,12,
                                          4,9,10,5,  9,14,15,10,  1,4, 5, 2,  11,12,15,14} );

  CHECK( penta_elt_type == cgns::PENTA_6 );
  CHECK( penta_range    == std::vector<I4>{15,16} );
  CHECK( penta_connec   == std::vector<I4>{3,5,2,8,10,7,  7,8,10,12,13,15} );

  CHECK( hexa_elt_type == cgns::HEXA_8 );
  CHECK( hexa_range    == std::vector<I4>{17,18} );
  CHECK( hexa_connec   == std::vector<I4>{2,5,4,1,7,10,9,6,  6,7,10,9,11,12,15,14} );
}
#endif // C++>17
