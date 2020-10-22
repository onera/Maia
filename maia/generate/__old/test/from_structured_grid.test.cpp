#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/structured_grid_utils.hpp"
#include "maia/generate/__old/from_structured_grid.hpp"
#include "range/v3/range/conversion.hpp"

using std::vector;
using cgns::quad_4;
using cgns::hex_8;

namespace doctest {
template<class I, class CK> struct StringMaker<connectivity<I,CK>> {
  static String convert(const connectivity<I,CK>& v) {
    std::string s = std_e::range_to_string(v);
    return s.c_str();
  }
};
} // doctest

TEST_CASE("test__generate_connectivities__one_quad") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {2,2,2};

  // cf. connectivities in simple_meshes.h
  hex_8<int> expected_cell = {0,1,3,2,4,5,7,6};
  
  quad_4<int> expected_i_face_0 = {0,2,6,4};
  quad_4<int> expected_i_face_1 = {1,3,7,5};

  quad_4<int> expected_j_face_0 = {0,4,5,1};
  quad_4<int> expected_j_face_1 = {2,6,7,3};

  quad_4<int> expected_k_face_0 = {0,1,3,2};
  quad_4<int> expected_k_face_1 = {4,5,7,6};

  vector<int> expected_r_parents = {{0,no_parent_element, 0,no_parent_element, 0,no_parent_element}};
  vector<int> expected_l_parents = {{no_parent_element,0, no_parent_element,0, no_parent_element,0}};

  SUBCASE("test__generate_connectivities__one_quad,structured_block_cell_dims") {
    REQUIRE( structured_block_cell_dims(vertex_dims) == MI{1,1,1} );
  }

  SUBCASE("generate one hexa") {
    auto cells = generate_cells(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( cells.size() == 1 );
    CHECK(cells[0] == expected_cell );
  }

  SUBCASE("generate i-faces of one hexa") {
    auto i_faces = generate_faces(vertex_dims,0) | ranges::to<std::vector>;

    REQUIRE( i_faces.size() == 2 );
    CHECK( i_faces[0] == expected_i_face_0 );
    CHECK( i_faces[1] == expected_i_face_1 );
  }

  SUBCASE("generate j-faces of one hexa") {
    auto j_faces = generate_faces(vertex_dims,1) | ranges::to<std::vector>;

    REQUIRE( j_faces.size() == 2 );
    CHECK( j_faces[0] == expected_j_face_0 );
    CHECK( j_faces[1] == expected_j_face_1 );
  }

  SUBCASE("generate k-faces of one hexa") {
    auto k_faces = generate_faces(vertex_dims,2) | ranges::to<std::vector>;

    REQUIRE( k_faces.size() == 2 );
    CHECK( k_faces[0] == expected_k_face_0 );
    CHECK( k_faces[1] == expected_k_face_1 );
  }


  SUBCASE("generate all faces of one hexa") {
    auto faces = generate_faces(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( faces.size() == 6 );
    CHECK( faces[0] == expected_i_face_0 );
    CHECK( faces[1] == expected_i_face_1 );
    CHECK( faces[2] == expected_j_face_0 );
    CHECK( faces[3] == expected_j_face_1 );
    CHECK( faces[4] == expected_k_face_0 );
    CHECK( faces[5] == expected_k_face_1 );
  }

  SUBCASE("generate right parents of one hexa") {
    vector<int> r_parents = generate_faces_right_parent_cell_ids(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( r_parents.size() == 6 );
    CHECK( r_parents[0] == expected_r_parents[0] );
    CHECK( r_parents[1] == expected_r_parents[1] );
    CHECK( r_parents[2] == expected_r_parents[2] );
    CHECK( r_parents[3] == expected_r_parents[3] );
    CHECK( r_parents[4] == expected_r_parents[4] );
    CHECK( r_parents[5] == expected_r_parents[5] );
  }

  SUBCASE("generate left parents of one hexa") {
    vector<int> l_parents = generate_faces_left_parent_cell_ids(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( l_parents.size() == 6 );
    CHECK( l_parents[0] == expected_l_parents[0] );
    CHECK( l_parents[1] == expected_l_parents[1] );
    CHECK( l_parents[2] == expected_l_parents[2] );
    CHECK( l_parents[3] == expected_l_parents[3] );
    CHECK( l_parents[4] == expected_l_parents[4] );
    CHECK( l_parents[5] == expected_l_parents[5] );
  }
};



TEST_CASE("test__generate_connectivities__six_quads") {
  using MI = std_e::multi_index<int,3>;
  MI vertex_dims = {4,3,2};

  // cf. connectivities in simple_meshes.h
  hex_8<int> expected_cell_0 = {0,1,5,4,12,13,17,16};
  hex_8<int> expected_cell_1 = {1,2,6,5,13,14,18,17};
  hex_8<int> expected_cell_2 = {2,3,7,6,14,15,19,18};
  hex_8<int> expected_cell_3 = {4,5,9,8,16,17,21,20};
  hex_8<int> expected_cell_4 = {5,6,10,9,17,18,22,21};
  hex_8<int> expected_cell_5 = {6,7,11,10,18,19,23,22};

  quad_4<int> expected_i_face_0 = {0,4,16,12};
  quad_4<int> expected_i_face_1 = {4,8,20,16};
  quad_4<int> expected_i_face_2 = {1,5,17,13};
  quad_4<int> expected_i_face_3 = {5,9,21,17};
  quad_4<int> expected_i_face_4 = {2,6,18,14};
  quad_4<int> expected_i_face_5 = {6,10,22,18};
  quad_4<int> expected_i_face_6 = {3,7,19,15};
  quad_4<int> expected_i_face_7 = {7,11,23,19};

  quad_4<int> expected_j_face_0 = {0,12,13,1};
  quad_4<int> expected_j_face_1 = {1,13,14,2};
  quad_4<int> expected_j_face_2 = {2,14,15,3};
  quad_4<int> expected_j_face_3 = {4,16,17,5};
  quad_4<int> expected_j_face_4 = {5,17,18,6};
  quad_4<int> expected_j_face_5 = {6,18,19,7};
  quad_4<int> expected_j_face_6 = {8,20,21,9};
  quad_4<int> expected_j_face_7 = {9,21,22,10};
  quad_4<int> expected_j_face_8 = {10,22,23,11};

  quad_4<int> expected_k_face_0 = {0,1,5,4};
  quad_4<int> expected_k_face_1 = {1,2,6,5};
  quad_4<int> expected_k_face_2 = {2,3,7,6};
  quad_4<int> expected_k_face_3 = {4,5,9,8};
  quad_4<int> expected_k_face_4 = {5,6,10,9};
  quad_4<int> expected_k_face_5 = {6,7,11,10};
  quad_4<int> expected_k_face_6 = {12,13,17,16};
  quad_4<int> expected_k_face_7 = {13,14,18,17};
  quad_4<int> expected_k_face_8 = {14,15,19,18};
  quad_4<int> expected_k_face_9 = {16,17,21,20};
  quad_4<int> expected_k_face_10 = {17,18,22,21};
  quad_4<int> expected_k_face_11 = {18,19,23,22};

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


  SUBCASE("test__generate_connectivities__six_quads,structured_block_cell_dims") {
    auto cell_dims = structured_block_cell_dims(vertex_dims);

    REQUIRE( cell_dims == MI{3,2,1} );
  }

  SUBCASE("generate six hexas") {
    auto cells = generate_cells(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( cells.size() == 6 );
    CHECK( cells[0] == expected_cell_0 );
    CHECK( cells[1] == expected_cell_1 );
    CHECK( cells[2] == expected_cell_2 );
    CHECK( cells[3] == expected_cell_3 );
    CHECK( cells[4] == expected_cell_4 );
    CHECK( cells[5] == expected_cell_5 );
  }

  SUBCASE("generate i-faces of six hexas") {
    auto i_faces = generate_faces(vertex_dims,0) | ranges::to<std::vector>;

    REQUIRE( i_faces.size() == 8 );
    CHECK( i_faces[0] == expected_i_face_0 );
    CHECK( i_faces[1] == expected_i_face_1 );
    CHECK( i_faces[2] == expected_i_face_2 );
    CHECK( i_faces[3] == expected_i_face_3 );
    CHECK( i_faces[4] == expected_i_face_4 );
    CHECK( i_faces[5] == expected_i_face_5 );
    CHECK( i_faces[6] == expected_i_face_6 );
    CHECK( i_faces[7] == expected_i_face_7 );
  }

  SUBCASE("generate j-faces of six hexas") {
    auto j_faces = generate_faces(vertex_dims,1) | ranges::to<std::vector>;

    REQUIRE( j_faces.size() == 9 );
    CHECK( j_faces[0] == expected_j_face_0 );
    CHECK( j_faces[1] == expected_j_face_1 );
    CHECK( j_faces[2] == expected_j_face_2 );
    CHECK( j_faces[3] == expected_j_face_3 );
    CHECK( j_faces[4] == expected_j_face_4 );
    CHECK( j_faces[5] == expected_j_face_5 );
    CHECK( j_faces[6] == expected_j_face_6 );
    CHECK( j_faces[7] == expected_j_face_7 );
    CHECK( j_faces[8] == expected_j_face_8 );
  }

  SUBCASE("generate k-faces of six hexas") {
    auto k_faces = generate_faces(vertex_dims,2) | ranges::to<std::vector>;

    REQUIRE( k_faces.size() == 12 );
    CHECK( k_faces [0] == expected_k_face_0  );
    CHECK( k_faces [1] == expected_k_face_1  );
    CHECK( k_faces [2] == expected_k_face_2  );
    CHECK( k_faces [3] == expected_k_face_3  );
    CHECK( k_faces [4] == expected_k_face_4  );
    CHECK( k_faces [5] == expected_k_face_5  );
    CHECK( k_faces [6] == expected_k_face_6  );
    CHECK( k_faces [7] == expected_k_face_7  );
    CHECK( k_faces [8] == expected_k_face_8  );
    CHECK( k_faces [9] == expected_k_face_9  );
    CHECK( k_faces[10] == expected_k_face_10 );
    CHECK( k_faces[11] == expected_k_face_11 );
  }

  SUBCASE("generate right parents of six hexas") {
    vector<int> r_parents = generate_faces_right_parent_cell_ids(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( r_parents.size() == 8+9+12 );
    CHECK( r_parents == expected_r_parents );
  }

  SUBCASE("generate left parents of six hexas") {
    vector<int> l_parents = generate_faces_left_parent_cell_ids(vertex_dims) | ranges::to<std::vector>;

    REQUIRE( l_parents.size() == 8+9+12 );
    CHECK( l_parents == expected_l_parents );
  }
};
