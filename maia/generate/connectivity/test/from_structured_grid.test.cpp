#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/connectivity/from_structured_grid.hpp"
#include "maia/utils/cgns_tree_examples/simple_meshes.hpp"

using std::array;
using namespace maia;
using namespace maia::six_hexa_mesh;

TEST_CASE("generate for maia::six_hexa_mesh") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {4,3,2};

  // Checks
  CHECK( generate_hex_8(vertex_dims,MI{0,0,0}) == cell_vtx[0] );
  CHECK( generate_hex_8(vertex_dims,MI{1,0,0}) == cell_vtx[1] );
  CHECK( generate_hex_8(vertex_dims,MI{2,0,0}) == cell_vtx[2] );
  CHECK( generate_hex_8(vertex_dims,MI{0,1,0}) == cell_vtx[3] );
  CHECK( generate_hex_8(vertex_dims,MI{1,1,0}) == cell_vtx[4] );
  CHECK( generate_hex_8(vertex_dims,MI{2,1,0}) == cell_vtx[5] );

  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{0,0,0}) == i_face_vtx[0] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{0,1,0}) == i_face_vtx[1] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{1,0,0}) == i_face_vtx[2] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{1,1,0}) == i_face_vtx[3] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{2,0,0}) == i_face_vtx[4] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{2,1,0}) == i_face_vtx[5] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{3,0,0}) == i_face_vtx[6] );
  CHECK( generate_quad_4_normal_to<0>(vertex_dims,MI{3,1,0}) == i_face_vtx[7] );

  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{0,0,0}) == j_face_vtx[0] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{1,0,0}) == j_face_vtx[1] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{2,0,0}) == j_face_vtx[2] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{0,1,0}) == j_face_vtx[3] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{1,1,0}) == j_face_vtx[4] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{2,1,0}) == j_face_vtx[5] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{0,2,0}) == j_face_vtx[6] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{1,2,0}) == j_face_vtx[7] );
  CHECK( generate_quad_4_normal_to<1>(vertex_dims,MI{2,2,0}) == j_face_vtx[8] );

  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{0,0,0}) == k_face_vtx[0]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{1,0,0}) == k_face_vtx[1]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{2,0,0}) == k_face_vtx[2]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{0,1,0}) == k_face_vtx[3]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{1,1,0}) == k_face_vtx[4]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{2,1,0}) == k_face_vtx[5]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{0,0,1}) == k_face_vtx[6]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{1,0,1}) == k_face_vtx[7]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{2,0,1}) == k_face_vtx[8]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{0,1,1}) == k_face_vtx[9]  );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{1,1,1}) == k_face_vtx[10] );
  CHECK( generate_quad_4_normal_to<2>(vertex_dims,MI{2,1,1}) == k_face_vtx[11] );
};
