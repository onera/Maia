#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/connectivity/from_structured_grid.hpp"

using std::array;
using namespace maia;

TEST_CASE("generate for one cell") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {2,2,2};

  // Expected values cf. connectivities in simple_meshes.hpp // TODO move to doc
  array<int,8> expected_cell = {0,1,3,2,4,5,7,6};

  array<int,4> expected_i_face_0 = {0,2,6,4};
  array<int,4> expected_i_face_1 = {1,3,7,5};

  array<int,4> expected_j_face_0 = {0,4,5,1};
  array<int,4> expected_j_face_1 = {2,6,7,3};

  array<int,4> expected_k_face_0 = {0,1,3,2};
  array<int,4> expected_k_face_1 = {4,5,7,6};

  // Checks
  MI only_cell_pos = {0,0,0};
  CHECK( generate_hex_8(vertex_dims,only_cell_pos) == expected_cell );

  MI i_face_0_pos = {0,0,0};
  MI i_face_1_pos = {1,0,0};
  CHECK( generate_quad_4_normal_to_i(vertex_dims,i_face_0_pos) == expected_i_face_0 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,i_face_1_pos) == expected_i_face_1 );

  MI j_face_0_pos = {0,0,0};
  MI j_face_1_pos = {0,1,0};
  CHECK( generate_quad_4_normal_to_j(vertex_dims,j_face_0_pos) == expected_j_face_0 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,j_face_1_pos) == expected_j_face_1 );

  MI k_face_0_pos = {0,0,0};
  MI k_face_1_pos = {0,0,1};
  CHECK( generate_quad_4_normal_to_k(vertex_dims,k_face_0_pos) == expected_k_face_0 );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,k_face_1_pos) == expected_k_face_1 );
};



TEST_CASE("generate for six cells") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {4,3,2};

  // Expected values cf. connectivities in simple_meshes.hpp // TODO move to doc
  array<int,8> expected_cell_0 = {0,1,5,4,12,13,17,16};
  array<int,8> expected_cell_1 = {1,2,6,5,13,14,18,17};
  array<int,8> expected_cell_2 = {2,3,7,6,14,15,19,18};
  array<int,8> expected_cell_3 = {4,5,9,8,16,17,21,20};
  array<int,8> expected_cell_4 = {5,6,10,9,17,18,22,21};
  array<int,8> expected_cell_5 = {6,7,11,10,18,19,23,22};

  array<int,4> expected_i_face_0 = {0,4,16,12};
  array<int,4> expected_i_face_1 = {4,8,20,16};
  array<int,4> expected_i_face_2 = {1,5,17,13};
  array<int,4> expected_i_face_3 = {5,9,21,17};
  array<int,4> expected_i_face_4 = {2,6,18,14};
  array<int,4> expected_i_face_5 = {6,10,22,18};
  array<int,4> expected_i_face_6 = {3,7,19,15};
  array<int,4> expected_i_face_7 = {7,11,23,19};

  array<int,4> expected_j_face_0 = {0,12,13,1};
  array<int,4> expected_j_face_1 = {1,13,14,2};
  array<int,4> expected_j_face_2 = {2,14,15,3};
  array<int,4> expected_j_face_3 = {4,16,17,5};
  array<int,4> expected_j_face_4 = {5,17,18,6};
  array<int,4> expected_j_face_5 = {6,18,19,7};
  array<int,4> expected_j_face_6 = {8,20,21,9};
  array<int,4> expected_j_face_7 = {9,21,22,10};
  array<int,4> expected_j_face_8 = {10,22,23,11};

  array<int,4> expected_k_face_0  = {0,1,5,4};
  array<int,4> expected_k_face_1  = {1,2,6,5};
  array<int,4> expected_k_face_2  = {2,3,7,6};
  array<int,4> expected_k_face_3  = {4,5,9,8};
  array<int,4> expected_k_face_4  = {5,6,10,9};
  array<int,4> expected_k_face_5  = {6,7,11,10};
  array<int,4> expected_k_face_6  = {12,13,17,16};
  array<int,4> expected_k_face_7  = {13,14,18,17};
  array<int,4> expected_k_face_8  = {14,15,19,18};
  array<int,4> expected_k_face_9  = {16,17,21,20};
  array<int,4> expected_k_face_10 = {17,18,22,21};
  array<int,4> expected_k_face_11 = {18,19,23,22};

  // Checks
  CHECK( generate_hex_8(vertex_dims,MI{0,0,0}) == expected_cell_0 );
  CHECK( generate_hex_8(vertex_dims,MI{1,0,0}) == expected_cell_1 );
  CHECK( generate_hex_8(vertex_dims,MI{2,0,0}) == expected_cell_2 );
  CHECK( generate_hex_8(vertex_dims,MI{0,1,0}) == expected_cell_3 );
  CHECK( generate_hex_8(vertex_dims,MI{1,1,0}) == expected_cell_4 );
  CHECK( generate_hex_8(vertex_dims,MI{2,1,0}) == expected_cell_5 );

  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{0,0,0}) == expected_i_face_0 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{0,1,0}) == expected_i_face_1 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{1,0,0}) == expected_i_face_2 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{1,1,0}) == expected_i_face_3 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{2,0,0}) == expected_i_face_4 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{2,1,0}) == expected_i_face_5 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{3,0,0}) == expected_i_face_6 );
  CHECK( generate_quad_4_normal_to_i(vertex_dims,MI{3,1,0}) == expected_i_face_7 );

  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{0,0,0}) == expected_j_face_0 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{1,0,0}) == expected_j_face_1 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{2,0,0}) == expected_j_face_2 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{0,1,0}) == expected_j_face_3 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{1,1,0}) == expected_j_face_4 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{2,1,0}) == expected_j_face_5 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{0,2,0}) == expected_j_face_6 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{1,2,0}) == expected_j_face_7 );
  CHECK( generate_quad_4_normal_to_j(vertex_dims,MI{2,2,0}) == expected_j_face_8 );

  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{0,0,0}) == expected_k_face_0  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{1,0,0}) == expected_k_face_1  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{2,0,0}) == expected_k_face_2  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{0,1,0}) == expected_k_face_3  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{1,1,0}) == expected_k_face_4  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{2,1,0}) == expected_k_face_5  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{0,0,1}) == expected_k_face_6  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{1,0,1}) == expected_k_face_7  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{2,0,1}) == expected_k_face_8  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{0,1,1}) == expected_k_face_9  );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{1,1,1}) == expected_k_face_10 );
  CHECK( generate_quad_4_normal_to_k(vertex_dims,MI{2,1,1}) == expected_k_face_11 );
};
