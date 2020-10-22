#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/ngons/from_cells/generate_faces_from_connectivity_elts.hpp"


TEST_CASE("generate_faces_from_connectivity_elts: homogenous_quad_connectivities") {
  std::vector<int> quads_data = {
    3,9,12,1,
    5,1, 2,4,
  };
  using Quad_kind = cgns::connectivity_kind<cgns::QUAD_4>;
  auto quads = make_connectivity_range<Quad_kind>(quads_data);
  
  auto faces = cgns::generate_faces_from_connectivity_elts(quads,101);

  // TODO explicit the fact that get<0> is tris and get<1> is quads
  CHECK( std_e::get<1>(faces.from_face).size() == 2 );

  auto quad_faces = std_e::get<1>(faces.from_face);
  CHECK( quad_faces.size() == 2 );

  quad_4_with_sorted_connectivity<int> expected_quad_faces_0 = {
    {3,9,12,1},
    {1,3,9,12},
    101
  };
  quad_4_with_sorted_connectivity<int> expected_quad_faces_1 = {
    {5,1,2,4},
    {1,2,4,5},
    102
  };
  CHECK( quad_faces[0] == expected_quad_faces_0 );
  CHECK( quad_faces[1] == expected_quad_faces_1 );
}

TEST_CASE("generate_faces_from_connectivity_elts: homogenous_tet_connectivities") {
  std::vector<int> tets_data = {
    3,9,12,1
  };
  using Tet_kind = cgns::connectivity_kind<cgns::TETRA_4>;
  auto tets = make_connectivity_range<Tet_kind>(tets_data);
  
  auto faces = cgns::generate_faces_from_connectivity_elts(tets,101);

// TODO explicit the fact that get<0> is tris and get<1> is quads
  CHECK( std_e::get<0>(faces.from_vol).size() == 4 );

  auto tri_faces = std_e::get<0>(faces.from_vol);

  tri_3_with_sorted_connectivity<int> expected_tri_faces_0 = {
    {3,12,9},
    {3,9,12},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_1 = {
    {3,9,1},
    {1,3,9},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_2 = {
    {3,1,12},
    {1,3,12},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_3 = {
    {9,12,1},
    {1,9,12},
    101
  };
  CHECK( tri_faces[0] == expected_tri_faces_0 );
  CHECK( tri_faces[1] == expected_tri_faces_1 );
  CHECK( tri_faces[2] == expected_tri_faces_2 );
  CHECK( tri_faces[3] == expected_tri_faces_3 );
}

TEST_CASE("generate_faces_from_connectivity_elts: heterogenous_tet_and_prism_connectivities") {
  std::vector<int> het_data = {
    cgns::TETRA_4,  3,9,12,1,
    cgns::PENTA_6, 12,9, 3,2,4,6
  };
  auto connectivities = cgns::interleaved_mixed_range(het_data);
  
  auto faces = cgns::generate_faces_from_connectivity_elts(connectivities,101);

  // TODO explicit the fact that get<0> is tris and get<1> is quads
  CHECK( std_e::get<0>(faces.from_vol).size() == 6 );
  CHECK( std_e::get<1>(faces.from_vol).size() == 3 );

  auto tri_faces = std_e::get<0>(faces.from_vol);
  auto quad_faces = std_e::get<1>(faces.from_vol);

  // tet
  tri_3_with_sorted_connectivity<int> expected_tri_faces_0 = {
    {3,12,9},
    {3,9,12},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_1 = {
    {3,9,1},
    {1,3,9},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_2 = {
    {3,1,12},
    {1,3,12},
    101
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_3 = {
    {9,12,1},
    {1,9,12},
    101
  };
  CHECK( tri_faces[0] == expected_tri_faces_0 );
  CHECK( tri_faces[1] == expected_tri_faces_1 );
  CHECK( tri_faces[2] == expected_tri_faces_2 );
  CHECK( tri_faces[3] == expected_tri_faces_3 );

  // prism
  tri_3_with_sorted_connectivity<int> expected_tri_faces_4 = {
    {12,3,9},
    {3,9,12},
    102
  };
  tri_3_with_sorted_connectivity<int> expected_tri_faces_5 = {
    {2,4,6},
    {2,4,6},
    102
  };
  quad_4_with_sorted_connectivity<int> expected_quad_faces_0 = {
    {12,9,4,2},
    {2,4,9,12},
    102
  };
  quad_4_with_sorted_connectivity<int> expected_quad_faces_1 = {
    {12,2,6,3},
    {2,3,6,12},
    102
  };
  quad_4_with_sorted_connectivity<int> expected_quad_faces_2 = {
    {9,3,6,4},
    {3,4,6,9},
    102
  };

  CHECK(  tri_faces[4] == expected_tri_faces_4  );
  CHECK(  tri_faces[5] == expected_tri_faces_5  );
  CHECK( quad_faces[0] == expected_quad_faces_0 );
  CHECK( quad_faces[1] == expected_quad_faces_1 );
  CHECK( quad_faces[2] == expected_quad_faces_2 );
}
