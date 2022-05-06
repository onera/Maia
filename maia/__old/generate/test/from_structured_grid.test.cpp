#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/__old/generate/structured_grid_utils.hpp"
#include "maia/__old/generate/from_structured_grid.hpp"
#include "maia/__old/utils/cgns_tree_examples/simple_meshes.hpp"

using std::vector;
using std::array;
using namespace maia;



TEST_CASE("generate for maia::six_hexa_mesh") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {4,3,2};

  auto cells = generate_cells(vertex_dims) | std_e::to_vector();
  auto faces = generate_faces(vertex_dims) | std_e::to_vector();
  auto l_parents = generate_l_parents(vertex_dims) | std_e::to_vector();
  auto r_parents = generate_r_parents(vertex_dims) | std_e::to_vector();

  CHECK( cells == maia::six_hexa_mesh::cell_vtx );
  CHECK( faces == maia::six_hexa_mesh::face_vtx );

  CHECK( l_parents == maia::six_hexa_mesh::l_parents );
  CHECK( r_parents == maia::six_hexa_mesh::r_parents );
};
#endif // C++>17
