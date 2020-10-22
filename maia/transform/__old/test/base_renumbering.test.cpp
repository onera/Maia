#include "doctest/extensions/doctest_mpi.h"

//#include "maia/sids_example/unstructured_base.hpp"
//#include "maia/transform/partition_with_boundary_first/partition_with_boundary_first.hpp"
//#include "std_e/algorithm/algorithm.hpp"
//#include "cpp_cgns/tree_manip.hpp"
//#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
//#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
//
//using namespace cgns;
//
//TEST_CASE("") {
//  cgns_allocator alloc; // allocates and owns memory
//  factory F(&alloc);
//
//  tree base = create_unstructured_base(F);
//  cgns::partition_with_boundary_first(base,F);
//  
//  // zone 0
//  tree& z0 = get_child_by_name(base,"Zone0");
//
//  // sizes
//  CHECK( VertexSize_U<I4>(z0) == 24 );
//  CHECK( CellSize_U<I4>(z0) == 6 );
//  CHECK( VertexBoundarySize_U<I4>(z0) == 20 );
//
//  // coordinates
//  tree& z0_coordX_node = get_node_by_matching(z0,"GridCoordinates/CoordinateX");
//  tree& z0_coordY_node = get_node_by_matching(z0,"GridCoordinates/CoordinateY");
//  tree& z0_coordZ_node = get_node_by_matching(z0,"GridCoordinates/CoordinateZ");
//  auto z0_coordX = view_as_span<R8>(z0_coordX_node.value);
//  auto z0_coordY = view_as_span<R8>(z0_coordY_node.value);
//  auto z0_coordZ = view_as_span<R8>(z0_coordZ_node.value);
//  std::vector<std::array<double,3>> expected_interior_elts_coords = {
//    {1.,1.,0}, // node 5 of simple_meshes.h
//    {2.,1.,0}, // node 6
//    {1.,1.,1}, // node 17
//    {2.,1.,1}, // node 18
//  };
//  std::array<double,3> interior_coord_0 = {z0_coordX[20],z0_coordY[20],z0_coordZ[20]};
//  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_0));
//
//  std::array<double,3> interior_coord_1 = {z0_coordX[21],z0_coordY[21],z0_coordZ[21]};
//  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_1));
//
//  std::array<double,3> interior_coord_2 = {z0_coordX[22],z0_coordY[22],z0_coordZ[22]};
//  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_2));
//
//  std::array<double,3> interior_coord_3 = {z0_coordX[23],z0_coordY[23],z0_coordZ[23]};
//  CHECK(std_e::contains(expected_interior_elts_coords,interior_coord_3));
//
//
//  // bcs
//  tree& z0_inflow_pl_node = get_node_by_matching(z0,"ZoneBC/Inlet/PointList");
//  auto z0_inflow_pl = view_as_span<I4>(z0_inflow_pl_node.value);
//  REQUIRE( z0_inflow_pl.size() == 2 );
//  CHECK( z0_inflow_pl[0] == 1 );
//  CHECK( z0_inflow_pl[1] == 2 );
//
//  // gcs
//  tree& z0_grid_connec_pl_node = get_node_by_matching(z0,"ZoneGridConnectivity/MixingPlane/PointList");
//  auto z0_grid_connec_pl = view_as_span<I4>(z0_grid_connec_pl_node.value);
//  tree& z0_grid_connec_pld_node = get_node_by_matching(z0,"ZoneGridConnectivity/MixingPlane/PointListDonor");
//  auto z0_grid_connec_pld = view_as_span<I4>(z0_grid_connec_pld_node.value);
//  REQUIRE( z0_grid_connec_pl.size() == 1 );
//  CHECK( z0_grid_connec_pl[0] == 3 );
//  REQUIRE( z0_grid_connec_pld.size() == 1 );
//  CHECK( z0_grid_connec_pld[0] == 1 );
//
//  // elements
//  tree& z0_ngon = get_child_by_name(z0,"Ngons");
//  auto z0_ngon_elt_range = ElementRange<I4>(z0_ngon);
//  CHECK( z0_ngon_elt_range[0] == 1 );
//  CHECK( z0_ngon_elt_range[1] == 8 + 9 + 12 );
//  auto& z0_nb_boundary_ngons = ElementSizeBoundary<I4>(z0_ngon);
//  CHECK( z0_nb_boundary_ngons == 3+2+3+2 );
//  auto z0_ngon_elt_connect = ElementConnectivity<I4>(z0_ngon);
//
//  REQUIRE( z0_ngon_elt_connect.size() == (8 + 9 + 12)*(1+4) );
//  CHECK( z0_ngon_elt_connect[0] ==  4 );
//  CHECK( z0_ngon_elt_connect[1] ==  1 );
//  CHECK( z0_ngon_elt_connect[2] ==  5 );
//  CHECK( z0_ngon_elt_connect[3] == 17 );
//  CHECK( z0_ngon_elt_connect[4] == 13 );
//  int beginning_last_ngon = (8+9+12-1)*(1+4);
//  CHECK( z0_ngon_elt_connect[beginning_last_ngon]   ==  4 );
//  CHECK( z0_ngon_elt_connect[beginning_last_ngon+1] == 21 ); // {21 20 6 7} are the new cgns vertex ids
//  CHECK( z0_ngon_elt_connect[beginning_last_ngon+2] == 20 ); // of face {18 19 23 22} in simple_meshes.h,
//  CHECK( z0_ngon_elt_connect[beginning_last_ngon+3] ==  6 ); // and this face is an interior face (because
//  CHECK( z0_ngon_elt_connect[beginning_last_ngon+4] ==  7 ); // in this test all k-faces are considered interior)
//                                                            
//  auto z0_ngon_parent_elts = ParentElements<I4>(z0_ngon);
//  REQUIRE( z0_ngon_parent_elts.size() == (8 + 9 + 12)*2 );
//  CHECK( z0_ngon_parent_elts(0,0) == 0 );  CHECK( z0_ngon_parent_elts(0,1) == 1 );
//  CHECK( z0_ngon_parent_elts(1,0) == 0 );  CHECK( z0_ngon_parent_elts(1,1) == 4 );
//  CHECK( z0_ngon_parent_elts(2,0) == 3 );  CHECK( z0_ngon_parent_elts(2,1) == 0 );
//  CHECK( z0_ngon_parent_elts(3,0) == 6 );  CHECK( z0_ngon_parent_elts(3,1) == 0 );
//}
