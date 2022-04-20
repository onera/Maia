#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"


#include "maia/__old/utils/cgns_tree_examples/unstructured_base.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
using namespace cgns;


TEST_CASE("unstructured mesh construction") {
  auto base = create_unstructured_base();

  // zone 0
  auto& z0 = get_child_by_label(base,"Zone_t");

  // sizes
  CHECK( VertexSize_U<I4>(z0) == 24 );
  CHECK( CellSize_U<I4>(z0) == 6 );
  CHECK( VertexBoundarySize_U<I4>(z0) == 0 );

  // coordinates
  tree& z0_coordX_node = get_node_by_matching(z0,"GridCoordinates/CoordinateX");
  auto z0_coordX = get_value<R8>(z0_coordX_node);
  std::vector<double> expected_z0_coord_X = {
    0.,1.,2.,3.,
    0.,1.,2.,3.,
    0.,1.,2.,3.,
    0.,1.,2.,3.,
    0.,1.,2.,3.,
    0.,1.,2.,3.
  };
  REQUIRE( z0_coordX.size() == expected_z0_coord_X.size() );
  for (size_t i=0; i<expected_z0_coord_X.size(); ++i) {
    CHECK( z0_coordX[i] == doctest::Approx(expected_z0_coord_X[i]) );
  }

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
  CHECK( z0_grid_connec_pl[0] == 7 );
  REQUIRE( z0_grid_connec_pld.size() == 1 );
  CHECK( z0_grid_connec_pld[0] == 1 );

  // elements
  tree& z0_ngon = get_child_by_name(z0,"Ngons");
  auto z0_ngon_elt_range = ElementRange<I4>(z0_ngon);
  CHECK( z0_ngon_elt_range[0] == 1 );
  CHECK( z0_ngon_elt_range[1] == 8 + 9 + 12 );
  auto z0_ngon_elt_connect = ElementConnectivity<I4>(z0_ngon);
  auto z0_ngon_eso = ElementStartOffset<I4>(z0_ngon);
  REQUIRE( z0_ngon_eso.size() == 8+9+12 + 1 );
  REQUIRE( z0_ngon_elt_connect.size() == (8 + 9 + 12)*4 );
  CHECK( z0_ngon_elt_connect[0] ==  1 );
  CHECK( z0_ngon_elt_connect[1] ==  5 );
  CHECK( z0_ngon_elt_connect[2] == 17 );
  CHECK( z0_ngon_elt_connect[3] == 13 );
  auto z0_ngon_parent_elts = ParentElements<I4>(z0_ngon);
  REQUIRE( z0_ngon_parent_elts.size() == (8 + 9 + 12)*2 );
  CHECK( z0_ngon_parent_elts(0,0) == 0 ); CHECK( z0_ngon_parent_elts(0,1) == 1 );
  CHECK( z0_ngon_parent_elts(1,0) == 0 ); CHECK( z0_ngon_parent_elts(1,1) == 4 );
  CHECK( z0_ngon_parent_elts(2,0) == 1 ); CHECK( z0_ngon_parent_elts(2,1) == 2 );
  CHECK( z0_ngon_parent_elts(3,0) == 4 ); CHECK( z0_ngon_parent_elts(3,1) == 5 );
}
#endif // C++>17
