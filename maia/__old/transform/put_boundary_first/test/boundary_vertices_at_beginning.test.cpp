#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/__old/transform/put_boundary_first/boundary_vertices_at_beginning.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"

using namespace std;
using namespace cgns;

TEST_CASE("vertex_permutation_to_move_boundary_at_beginning") {
  const I4 n_vtx = 10;

  std::vector<I4> boundary_vertex_ids = { 1, 2, 5, 9 };
  auto vertex_permutation = maia::vertex_permutation_to_move_boundary_at_beginning(n_vtx,boundary_vertex_ids);

  int partition_point = 4;
  std::vector<I4> boundary_vertex_positions = { 0, 1, 4, 8 }; // == boundary_vertex_ids - 1
  std::vector<I4> interior_vertex_positions = { 2, 3, 5, 6, 7, 9 }; // remaining positions

  REQUIRE( vertex_permutation.size() == n_vtx );
  for (int i=0; i<n_vtx; ++i) {
    if (i<partition_point) {
      CHECK( std_e::contains(boundary_vertex_positions,vertex_permutation[i]) );
    } else {
      CHECK( std_e::contains(interior_vertex_positions,vertex_permutation[i]) );
    }
  }
}


TEST_CASE("re_number_vertex_ids_in_elements") {
  vector<I4> my_vertex_permutation = { 0,1,     5,6,7,    2,3,4,     8,9 };
  //                                   |_|      |___|     |___|      |_|
  //                                  still       ^         ^       still
  //                                  same        |_________|       same
  //                                  position      rotated         position
  //
  //   so                            { 1,2,     3,4,5,    6,7,8,     9,10 };
  //   should become                 { 1,2,     6,7,8,    3,4,5      9,10 };

  SUBCASE("tris") {
    std::vector<I4> tri_cs =
      { 1, 2, 3,
        1, 3, 4,
        4, 5, 9  };
    tree tris = new_Elements(
      "Tri",
      cgns::TRI_3,
      std::move(tri_cs),
      1,3
    );
    maia::re_number_vertex_ids_in_elements(tris,my_vertex_permutation);

    auto tri_view = ElementConnectivity<I4>(tris);

    CHECK_EQ( 1 , tri_view[0] );
    CHECK_EQ( 2 , tri_view[1] );
    CHECK_EQ( 6 , tri_view[2] );

    CHECK_EQ( 1 , tri_view[3] );
    CHECK_EQ( 6 , tri_view[4] );
    CHECK_EQ( 7 , tri_view[5] );

    CHECK_EQ( 7 , tri_view[6] );
    CHECK_EQ( 8 , tri_view[7] );
    CHECK_EQ( 9 , tri_view[8] );
  }

  SUBCASE("ngons") {
    std::vector<I4> ngon_cs =
      { 1, 2, 3,
        4, 5, 6,10,
        9, 8, 7,
        1, 8, 9 };
    std::vector<I4> ngon_eso = {0,3,7,10,13};

    tree ngons = new_NgonElements(
      "Ngons",
      std::move(ngon_cs),
      6,9
    );
    emplace_child(ngons,new_DataArray("ElementStartOffset", node_value(std::move(ngon_eso))));

    maia::re_number_vertex_ids_in_elements(ngons,my_vertex_permutation);

    std::vector<I4> expected_ngon_cs =
      { 1, 2, 6,
        7, 8, 3,10,
        9, 5, 4,
        1, 5, 9 };
    CHECK( ElementConnectivity<I4>(ngons) == expected_ngon_cs );
  }
}
#endif // C++>17
