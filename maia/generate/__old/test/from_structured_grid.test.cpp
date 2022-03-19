#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/structured_grid_utils.hpp"
#include "maia/generate/__old/from_structured_grid.hpp"

using std::vector;
using std::array;
using namespace maia;


TEST_CASE("generate for one cell") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {2,2,2};

  // cf. connectivities in simple_meshes.h
  array<int,8> expected_cell = {0,1,3,2,4,5,7,6};

  array<int,4> expected_i_face_0 = {0,2,6,4};
  array<int,4> expected_i_face_1 = {1,3,7,5};

  array<int,4> expected_j_face_0 = {0,4,5,1};
  array<int,4> expected_j_face_1 = {2,6,7,3};

  array<int,4> expected_k_face_0 = {0,1,3,2};
  array<int,4> expected_k_face_1 = {4,5,7,6};

  vector<int> expected_r_parents = {{0,no_parent_element, 0,no_parent_element, 0,no_parent_element}};
  vector<int> expected_l_parents = {{no_parent_element,0, no_parent_element,0, no_parent_element,0}};

  SUBCASE("generate all faces of one hexa") {
    auto faces = generate_faces(vertex_dims);

    REQUIRE( faces.size() == 6 );
    CHECK( faces[0] == expected_i_face_0 );
    CHECK( faces[1] == expected_i_face_1 );
    CHECK( faces[2] == expected_j_face_0 );
    CHECK( faces[3] == expected_j_face_1 );
    CHECK( faces[4] == expected_k_face_0 );
    CHECK( faces[5] == expected_k_face_1 );
  }

  SUBCASE("generate right parents of one hexa") {
    vector<int> r_parents = generate_faces_right_parent_cell_ids(vertex_dims);

    REQUIRE( r_parents.size() == 6 );
    CHECK( r_parents[0] == expected_r_parents[0] );
    CHECK( r_parents[1] == expected_r_parents[1] );
    CHECK( r_parents[2] == expected_r_parents[2] );
    CHECK( r_parents[3] == expected_r_parents[3] );
    CHECK( r_parents[4] == expected_r_parents[4] );
    CHECK( r_parents[5] == expected_r_parents[5] );
  }

  SUBCASE("generate left parents of one hexa") {
    vector<int> l_parents = generate_faces_left_parent_cell_ids(vertex_dims);

    REQUIRE( l_parents.size() == 6 );
    CHECK( l_parents[0] == expected_l_parents[0] );
    CHECK( l_parents[1] == expected_l_parents[1] );
    CHECK( l_parents[2] == expected_l_parents[2] );
    CHECK( l_parents[3] == expected_l_parents[3] );
    CHECK( l_parents[4] == expected_l_parents[4] );
    CHECK( l_parents[5] == expected_l_parents[5] );
  }
};



TEST_CASE("generate for six cells") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {4,3,2};

  vector<int> expected_r_parents = {{
    0,3, 1,4, 2,5, no_parent_element,no_parent_element,
    0,1,2, 3,4,5, no_parent_element,no_parent_element,no_parent_element,
    0,1,2,3,4,5, no_parent_element,no_parent_element,no_parent_element,no_parent_element,no_parent_element,no_parent_element
  }};
  vector<int> expected_l_parents = {{
    no_parent_element,no_parent_element, 0,3, 1,4, 2,5,
    no_parent_element,no_parent_element,no_parent_element, 0,1,2, 3,4,5,
    no_parent_element,no_parent_element,no_parent_element,no_parent_element,no_parent_element,no_parent_element, 0,1,2,3,4,5
  }};


  SUBCASE("generate right parents of six hexas") {
    vector<int> r_parents = generate_faces_right_parent_cell_ids(vertex_dims);

    REQUIRE( r_parents.size() == 8+9+12 );
    CHECK( r_parents == expected_r_parents );
  }

  SUBCASE("generate left parents of six hexas") {
    vector<int> l_parents = generate_faces_left_parent_cell_ids(vertex_dims);

    REQUIRE( l_parents.size() == 8+9+12 );
    CHECK( l_parents == expected_l_parents );
  }
};
