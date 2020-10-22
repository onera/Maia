#include "maia/utils/cgns_tree_examples/base_two_ranks.hpp"

using namespace cgns;

namespace example {

auto
create_base_two_ranks(int mpi_rank, factory F) -> tree {
  STD_E_ASSERT(mpi_rank==0 || mpi_rank==1);
  tree b = F.newCGNSBase("Base",3,3);

/* Case (note that GridConnectivities are *not* symmetric:
  proc 0
  /Base/Zone0
  /Base/Zone0/ZGC/j0 --> Zone0
  /Base/Zone0/ZGC/j1 --> Zone1
  /Base/Zone3
  /Base/Zone3/ZGC/j2 --> Zone1
  /Base/Zone3/ZGC/j3 --> Zone1
 
  proc 1
  /Base/Zone1
  /Base/Zone1/ZGC/j4 --> Zone1
  /Base/Zone1/ZGC/j5 --> Zone0
  /Base/Zone2
  /Base/Zone2/ZGC/j6 --> Zone3
*/

  if (mpi_rank == 0) {
  // Zone0
    tree z0 = F.newUnstructuredZone("Zone0");

    auto pld00_data = make_cgns_vector( {1,2,3} , F.alloc() );
    tree pld00 = F.newPointList("PointListDonor",std_e::make_span(pld00_data));
    tree gc00 = F.newGridConnectivity("Join0","Zone0","FaceCenter","Abutting1to1");
    emplace_child(gc00,std::move(pld00));

    auto pld01_data = make_cgns_vector( {11,12,13,14} , F.alloc() );
    tree pld01 = F.newPointList("PointListDonor",std_e::make_span(pld01_data));
    tree gc01 = F.newGridConnectivity("Join1","Zone1","FaceCenter","Abutting1to1");
    emplace_child(gc01,std::move(pld01));

    tree zone_gc0 = F.newZoneGridConnectivity();
    emplace_child(zone_gc0,std::move(gc00));
    emplace_child(zone_gc0,std::move(gc01));
    emplace_child(z0,std::move(zone_gc0));

    emplace_child(b,std::move(z0));
  // Zone3
    tree z3 = F.newUnstructuredZone("Zone3");

    auto pld31a_data = make_cgns_vector( {15} , F.alloc() );
    tree pld31a = F.newPointList("PointListDonor",std_e::make_span(pld31a_data));
    tree gc31a = F.newGridConnectivity("Join2","Zone1","Vertex","Abutting1to1");
    emplace_child(gc31a,std::move(pld31a));

    auto pld31b_data = make_cgns_vector( {16,17} , F.alloc() );
    tree pld31b = F.newPointList("PointListDonor",std_e::make_span(pld31b_data));
    tree gc31b = F.newGridConnectivity("Join3","Zone1","Vertex","Abutting1to1");
    emplace_child(gc31b,std::move(pld31b));

    tree zone_gc3 = F.newZoneGridConnectivity();
    emplace_child(zone_gc3,std::move(gc31a));
    emplace_child(zone_gc3,std::move(gc31b));
    emplace_child(z3,std::move(zone_gc3));

    emplace_child(b,std::move(z3));
  } else { STD_E_ASSERT(mpi_rank == 1); 
  // Zone1
    tree z1 = F.newUnstructuredZone("Zone1");

    auto pld11_data = make_cgns_vector( {101,102,103,104} , F.alloc() );
    tree pld11 = F.newPointList("PointListDonor",std_e::make_span(pld11_data));
    tree gc11 = F.newGridConnectivity("Join4","Zone1","CellCenter","Abutting1to1");
    emplace_child(gc11,std::move(pld11));

    auto pld10_data = make_cgns_vector( {111,112} , F.alloc() );
    tree pld10 = F.newPointList("PointListDonor",std_e::make_span(pld10_data));
    tree gc10 = F.newGridConnectivity("Join5","Zone0","Vertex","Abutting1to1");
    emplace_child(gc10,std::move(pld10));

    tree zone_gc1 = F.newZoneGridConnectivity();
    emplace_child(zone_gc1,std::move(gc11));
    emplace_child(zone_gc1,std::move(gc10));
    emplace_child(z1,std::move(zone_gc1));

    emplace_child(b,std::move(z1));

  // Zone2
    tree z2 = F.newUnstructuredZone("Zone2");

    auto pld21_data = make_cgns_vector( {136,137} , F.alloc() );
    tree pld21 = F.newPointList("PointListDonor",std_e::make_span(pld21_data));
    tree gc21 = F.newGridConnectivity("Join6","Zone3","CellCenter","Abutting1to1");
    emplace_child(gc21,std::move(pld21));

    tree zone_gc2 = F.newZoneGridConnectivity();
    emplace_child(zone_gc2,std::move(gc21));
    emplace_child(z2,std::move(zone_gc2));

    emplace_child(b,std::move(z2));
  }
  return b;
}

} // example
