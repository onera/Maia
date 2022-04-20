#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind.hpp"

#include "maia/__unused/flip_faces/flip_faces.hpp"
#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"

#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"

using namespace cgns;

PYBIND_TEST_CASE("flip_faces") {
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_prism.yaml");
  tree b = maia::to_node(yaml_tree);

  auto tris = ElementConnectivity<I4>(get_node_by_matching(b,"Zone/Tris"));
  auto quads = ElementConnectivity<I4>(get_node_by_matching(b,"Zone/Quads"));

  CHECK( tris == std::vector{2,5,3, 7,8,10} );
  CHECK( quads == std::vector{1,6,9,4, 3,5,10,8, 1,2,7,6, 2,3,8,7, 4,9,10,5, 1,4,5,2, 6,7,10,9} );

  maia::flip_faces(get_child_by_name(b,"Zone"));

  CHECK( tris == std::vector{3,5,2, 10,8,7} );
  CHECK( quads == std::vector{4,9,6,1, 8,10,5,3, 6,7,2,1, 7,8,3,2, 5,10,9,4, 2,5,4,1, 9,10,7,6} );
}
#endif // C++>17
