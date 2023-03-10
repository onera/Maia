#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/__old/utils/cgns_tree_examples/unstructured_base.hpp"
#include "maia/__old/transform/put_boundary_first/put_boundary_first.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"

using namespace cgns;

TEST_CASE("put_boundary_first, with 2 zones") {
  tree base = create_unstructured_base();

  maia::put_boundary_first(base,MPI_COMM_SELF);

  // zones
  tree& z0 = get_child_by_name(base,"Zone0");
  tree& z1 = get_child_by_name(base,"Zone1");

  // sizes
  CHECK( VertexSize_U<I4>(z0) == 24 );
  CHECK( CellSize_U<I4>(z0) == 6 );
  CHECK( VertexBoundarySize_U<I4>(z0) == 20 );

  // coordinates
  auto z0_coordX = get_node_value_by_matching<R8>(z0,"GridCoordinates/CoordinateX");
  auto z0_coordY = get_node_value_by_matching<R8>(z0,"GridCoordinates/CoordinateY");
  auto z0_coordZ = get_node_value_by_matching<R8>(z0,"GridCoordinates/CoordinateZ");
  std::vector<std::array<double,3>> expected_interior_elts_coords = {
    {1.,1.,0}, // node 5 of simple_meshes.h
    {2.,1.,0}, // node 6
    {1.,1.,1}, // node 17
    {2.,1.,1}, // node 18
  };
  std::array<double,3> interior_coord_0 = {z0_coordX[20],z0_coordY[20],z0_coordZ[20]};
  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_0));

  std::array<double,3> interior_coord_1 = {z0_coordX[21],z0_coordY[21],z0_coordZ[21]};
  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_1));

  std::array<double,3> interior_coord_2 = {z0_coordX[22],z0_coordY[22],z0_coordZ[22]};
  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_2));

  std::array<double,3> interior_coord_3 = {z0_coordX[23],z0_coordY[23],z0_coordZ[23]};
  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_3));


  // bcs
  tree& z0_inflow_pl_node = get_node_by_matching(z0,"ZoneBC/Inlet/PointList");
  auto z0_inflow_pl = get_value<I4>(z0_inflow_pl_node);
  REQUIRE( z0_inflow_pl.size() == 2 );
  CHECK( z0_inflow_pl[0] == 1 );
  CHECK( z0_inflow_pl[1] == 2 );

  // gcs
  tree& z0_grid_connec_pl_node = get_node_by_matching(z0,"ZoneGridConnectivity/MixingPlane/PointList");
  auto z0_grid_connec_pl = get_value<I4>(z0_grid_connec_pl_node);
  tree& z0_grid_connec_pld_node = get_node_by_matching(z0,"ZoneGridConnectivity/MixingPlane/PointListDonor");
  auto z0_grid_connec_pld = get_value<I4>(z0_grid_connec_pld_node);
  REQUIRE( z0_grid_connec_pl.size() == 1 );
  CHECK( z0_grid_connec_pl[0] == 3 );
  REQUIRE( z0_grid_connec_pld.size() == 1 );
  CHECK( z0_grid_connec_pld[0] == 1 );
  /// since there is no renumbering, in this case, on z0,
  /// we check the renumbering was done by looking z1
  tree& z1_grid_connec_pld_node = get_node_by_matching(z1,"ZoneGridConnectivity/MixingPlane/PointListDonor");
  auto z1_grid_connec_pld = get_value<I4>(z1_grid_connec_pld_node);
  REQUIRE( z1_grid_connec_pld.size() == 1 );
  //CHECK( z1_grid_connec_pld[0] == 3 ); // TODO fails gcc 10

  // elements
  tree& z0_ngon = get_child_by_name(z0,"Ngons");
  auto z0_ngon_elt_range = ElementRange<I4>(z0_ngon);
  CHECK( z0_ngon_elt_range[0] == 1 );
  CHECK( z0_ngon_elt_range[1] == 8 + 9 + 12 );
  auto z0_nb_boundary_ngons = ElementSizeBoundary(z0_ngon);
  CHECK( z0_nb_boundary_ngons == 3+2+3+2 );
  auto z0_ngon_elt_connect = ElementConnectivity<I4>(z0_ngon);

  REQUIRE( z0_ngon_elt_connect.size() == (8 + 9 + 12)*4 );
  CHECK( z0_ngon_elt_connect[0] ==  1 );
  CHECK( z0_ngon_elt_connect[1] ==  5 );
  CHECK( z0_ngon_elt_connect[2] == 17 );
  CHECK( z0_ngon_elt_connect[3] == 13 );
  int beginning_last_ngon = (8+9+12-1)*4;
  CHECK( z0_ngon_elt_connect[beginning_last_ngon+0] == 21 ); // {21 20 6 7} are the new cgns vertex ids
  CHECK( z0_ngon_elt_connect[beginning_last_ngon+1] == 20 ); // of face {18 19 23 22} in simple_meshes.h,
  CHECK( z0_ngon_elt_connect[beginning_last_ngon+2] ==  6 ); // and this face is an interior face (because
  CHECK( z0_ngon_elt_connect[beginning_last_ngon+3] ==  7 ); // in this test all k-faces are considered interior)

  auto z0_ngon_parent_elts = ParentElements<I4>(z0_ngon);
  REQUIRE( z0_ngon_parent_elts.size() == (8 + 9 + 12)*2 );
  CHECK( z0_ngon_parent_elts(0,0) == 0 );  CHECK( z0_ngon_parent_elts(0,1) == 1 );
  CHECK( z0_ngon_parent_elts(1,0) == 0 );  CHECK( z0_ngon_parent_elts(1,1) == 4 );
  CHECK( z0_ngon_parent_elts(2,0) == 3 );  CHECK( z0_ngon_parent_elts(2,1) == 0 );
  CHECK( z0_ngon_parent_elts(3,0) == 6 );  CHECK( z0_ngon_parent_elts(3,1) == 0 );
}
#endif // C++>17
