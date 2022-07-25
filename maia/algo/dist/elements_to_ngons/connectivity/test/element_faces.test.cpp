#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/algo/dist/elements_to_ngons/connectivity/element_faces.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"
#include "cpp_cgns/sids.hpp"

using namespace cgns;
using namespace maia;
using std::array;

// SEE https://cgns.github.io/CGNS_docs_current/sids/conv.html#unst_3d
TEST_CASE("generate_faces") {
  std::vector<int> gen_tris;
  auto tri_range = std_e::view_as_block_range<number_of_vertices(TRI_3)>(gen_tris);
  auto tri_back = back_inserter(tri_range);

  std::vector<int> gen_quads;
  auto quad_range = std_e::view_as_block_range<number_of_vertices(QUAD_4)>(gen_quads);
  auto quad_back = back_inserter(quad_range);

  std::vector<int> tri_pos;
  auto tri_pos_back = back_inserter(tri_pos);

  std::vector<int> quad_pos;
  auto quad_pos_back = back_inserter(quad_pos);

  SUBCASE("tri_3") {
    array<int,3> tri = {1,2,3};

    generate_faces<TRI_3>(tri,tri_back,quad_back);
    generate_parent_positions<TRI_3>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector{1,2,3} );
    CHECK( gen_quads == std::vector<int>{} );

    CHECK( tri_pos  == std::vector{1} );
    CHECK( quad_pos == std::vector<int>{} );
  }
  SUBCASE("quad_4") {
    array<int,4> quad = {1,2,3,4};

    generate_faces<QUAD_4>(quad,tri_back,quad_back);
    generate_parent_positions<QUAD_4>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector<int>{} );
    CHECK( gen_quads == std::vector{1,2,3,4} );

    CHECK( tri_pos  == std::vector<int>{} );
    CHECK( quad_pos == std::vector{1} );
  }
  SUBCASE("tetra_4") {
    array<int,4> tet = {1,2,3,4};

    generate_faces<TETRA_4>(tet,tri_back,quad_back);
    generate_parent_positions<TETRA_4>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector{1,3,2, 1,2,4, 2,3,4, 3,1,4} );
    CHECK( gen_quads == std::vector<int>{} );

    CHECK( tri_pos  == std::vector{1,2,3,4} );
    CHECK( quad_pos == std::vector<int>{} );
  }
  SUBCASE("pyra_5") {
    array<int,5> pyra = {1,2,3,4,5};

    generate_faces<PYRA_5>(pyra,tri_back,quad_back);
    generate_parent_positions<PYRA_5>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector{1,2,5, 2,3,5, 3,4,5, 4,1,5} );
    CHECK( gen_quads == std::vector{1,4,3,2} );

    CHECK( tri_pos  == std::vector{2,3,4,5} );
    CHECK( quad_pos == std::vector{1} );
  }
  SUBCASE("penta_6") {
    array<int,6> penta = {1,2,3,4,5,6};

    generate_faces<PENTA_6>(penta,tri_back,quad_back);
    generate_parent_positions<PENTA_6>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector{1,3,2, 4,5,6} );
    CHECK( gen_quads == std::vector{1,2,5,4, 2,3,6,5, 3,1,4,6} );

    CHECK( tri_pos  == std::vector{4,5} );
    CHECK( quad_pos == std::vector{1,2,3} );
  }
  SUBCASE("hexa_8") {
    array<int,8> hex = {1,2,3,4,5,6,7,8};

    generate_faces<HEXA_8>(hex,tri_back,quad_back);
    generate_parent_positions<HEXA_8>(tri_pos_back,quad_pos_back);

    CHECK( gen_tris  == std::vector<int>{} );
    CHECK( gen_quads == std::vector{1,4,3,2, 1,2,6,5, 2,3,7,6, 3,4,8,7, 1,5,8,4, 5,6,7,8} );

    CHECK( tri_pos  == std::vector<int>{} );
    CHECK( quad_pos == std::vector{1,2,3,4,5,6} );
  }
}

TEST_CASE("generate_faces: example with several connec") {
  // prepare
  std::vector<int> gen_tris(6);
  auto tri_range = std_e::view_as_block_range<number_of_vertices(TRI_3)>(gen_tris);
  auto tri_it = tri_range.begin();

  std::vector<int> gen_quads(8);
  auto quad_range = std_e::view_as_block_range<number_of_vertices(QUAD_4)>(gen_quads);
  auto quad_it = quad_range.begin();

  std::vector<int> tri_pos(2);
  auto tri_pos_it = tri_pos.begin();

  std::vector<int> quad_pos(2);
  auto quad_pos_it = quad_pos.begin();

  // gen several connec
  array<int,3> tri0  = {10,11,12};
  generate_faces<TRI_3 >(tri0 ,tri_it,quad_it);
  generate_parent_positions<TRI_3>(tri_pos_it,quad_pos_it);

  array<int,4> quad0 = {1,2,3,4};
  generate_faces<QUAD_4>(quad0,tri_it,quad_it);
  generate_parent_positions<QUAD_4>(tri_pos_it,quad_pos_it);

  array<int,4> quad1 = {4,3,2,1};
  generate_faces<QUAD_4>(quad1,tri_it,quad_it);
  generate_parent_positions<QUAD_4>(tri_pos_it,quad_pos_it);

  array<int,3> tri1  = {20,21,22};
  generate_faces<TRI_3 >(tri1 ,tri_it,quad_it);
  generate_parent_positions<TRI_3>(tri_pos_it,quad_pos_it);

  CHECK( gen_tris  == std::vector{10,11,12, 20,21,22} );
  CHECK( gen_quads == std::vector{1,2,3,4, 4,3,2,1} );
  CHECK( tri_pos == std::vector{1,1} );
  CHECK( quad_pos == std::vector{1,1} );
}
#endif // C++>17
