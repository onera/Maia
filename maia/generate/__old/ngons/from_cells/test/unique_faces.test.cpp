#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/ngons/from_cells/unique_faces.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"

using test__face_type = face_with_sorted_connectivity<int32_t,cgns::connectivity_kind<cgns::QUAD_4>>;

TEST_CASE("convert_to_interior_or_boundary_face") {
  test__face_type quad0 = {
    {1,2,4,3}, // connectivity
    {1,2,3,4}, // sorted connectivity
    42 // parent element
  };

  test__face_type quad1 = {
    {3,4,2,1},
    {1,2,3,4},
    43
  }; // same as quad0, but reversed face

  REQUIRE(same_face(quad0,quad1));

  auto boundary_quad = convert_to_boundary_face(quad0);
  CHECK( boundary_quad.connec == cgns::quad_4<int>{1,2,4,3} );
  CHECK( boundary_quad.l_parent == 42 );

  auto interior_quad = convert_to_interior_face(quad0,quad1);
  CHECK( interior_quad.connec == cgns::quad_4<int>{1,2,4,3} );
  CHECK( interior_quad.l_parent == 42 );
  CHECK( interior_quad.r_parent == 43 );

  auto interior_quad_reverse = convert_to_interior_face(quad1,quad0);
  CHECK( interior_quad_reverse.connec == cgns::quad_4<int>{3,4,2,1} );
  CHECK( interior_quad_reverse.l_parent == 43 );
  CHECK( interior_quad_reverse.r_parent == 42 );
}

// TODO TEST_CASE("unique_faces)

TEST_CASE("append_boundary_and_interior_faces") {
  const std::vector<int> het_data = {
    cgns::TETRA_4,  3,9,12,1,
    cgns::PENTA_6, 12,9,3,2,4,6,
    cgns::TRI_3,  12,1,3
  };
  auto connectivities = cgns::interleaved_mixed_range(het_data);

  int first_elt_id = 101;

  auto faces = cgns::generate_faces_from_connectivity_elts(connectivities,first_elt_id);

  faces_container<int> all_faces;
  // TODO merge with what was done with unique_compress
  //cgns::append_boundary_and_interior_faces(all_faces,faces);

  //// TODO explicit the fact that get<0> is tris and get<1> is quads
  //auto&  tri_bnd = std_e::get<0>(all_faces.boundary);
  //auto&  tri_int = std_e::get<0>(all_faces.interior);
  //auto& quad_bnd = std_e::get<1>(all_faces.boundary);
  //auto& quad_int = std_e::get<1>(all_faces.interior);

  //CHECK(  tri_bnd.size() == 4 );
  //CHECK(  tri_int.size() == 1 );
  //CHECK( quad_bnd.size() == 3 );
  //CHECK( quad_int.size() == 0 );

  //boundary_tri_3<int> expected_tri_bnd_0 = { // first because one of its parent is a face
  //  {12,1,3}, // TODO note: This is the connectivity of the parent face.
  //  103       //            It would be better for it to be the connectivity of the volume parent
  //};          //            So that the face parent will mean the right parent 
  //            //            (currently, could be left or right, depending on the face parent which 
  //            //            can't possibly know where is the interior)
  //boundary_tri_3<int> expected_tri_bnd_1 = {
  //  {3,9,1},
  //  101
  //};
  //boundary_tri_3<int> expected_tri_bnd_2 = {
  //  {9,12,1},
  //  101
  //};
  //boundary_tri_3<int> expected_tri_bnd_3 = {
  //  {2,4,6},
  //  102
  //};
  //CHECK( tri_bnd[0] == expected_tri_bnd_0 );
  //CHECK( tri_bnd[1] == expected_tri_bnd_1 );
  //CHECK( tri_bnd[2] == expected_tri_bnd_2 );
  //CHECK( tri_bnd[3] == expected_tri_bnd_3 );


  //interior_tri_3<int> expected_tri_int_0 = {
  //  {3,12,9},
  //  101,
  //  102
  //};
  //CHECK(tri_int[0] == expected_tri_int_0 );


  //boundary_quad_4<int> expected_quad_bnd_0 = {
  //  {12,2,6,3},
  //  102
  //};
  //boundary_quad_4<int> expected_quad_bnd_1 = {
  //  {12,9,4,2},
  //  102
  //};
  //boundary_quad_4<int> expected_quad_bnd_2 = {
  //  {9,3,6,4},
  //  102
  //};
  //CHECK( quad_bnd[0] == expected_quad_bnd_0 );
  //CHECK( quad_bnd[1] == expected_quad_bnd_1 );
  //CHECK( quad_bnd[2] == expected_quad_bnd_2 );
}
