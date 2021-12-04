#include "maia/utils/cgns_tree_examples/base_two_ranks.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;

namespace example {

auto
create_base_two_ranks(int mpi_rank) -> tree {
  STD_E_ASSERT(mpi_rank==0 || mpi_rank==1);
  tree b = new_CGNSBase("Base",3,3);

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
    tree z0 = new_UnstructuredZone<I4>("Zone0");

    tree pld00 = new_PointList("PointListDonor",std::vector{1,2,3});
    tree gc00 = new_GridConnectivity("Join0","Zone0","FaceCenter","Abutting1to1");
    emplace_child(gc00,std::move(pld00));

    tree pld01 = new_PointList("PointListDonor",std::vector{11,12,13,14});
    tree gc01 = new_GridConnectivity("Join1","Zone1","FaceCenter","Abutting1to1");
    emplace_child(gc01,std::move(pld01));

    tree zone_gc0 = new_ZoneGridConnectivity();
    emplace_child(zone_gc0,std::move(gc00));
    emplace_child(zone_gc0,std::move(gc01));
    emplace_child(z0,std::move(zone_gc0));

    emplace_child(b,std::move(z0));
  // Zone3
    tree z3 = new_UnstructuredZone<I4>("Zone3");

    tree pld31a = new_PointList("PointListDonor",std::vector{15});
    tree gc31a = new_GridConnectivity("Join2","Zone1","Vertex","Abutting1to1");
    emplace_child(gc31a,std::move(pld31a));

    tree pld31b = new_PointList("PointListDonor",std::vector{16,17});
    tree gc31b = new_GridConnectivity("Join3","Zone1","Vertex","Abutting1to1");
    emplace_child(gc31b,std::move(pld31b));

    tree zone_gc3 = new_ZoneGridConnectivity();
    emplace_child(zone_gc3,std::move(gc31a));
    emplace_child(zone_gc3,std::move(gc31b));
    emplace_child(z3,std::move(zone_gc3));

    emplace_child(b,std::move(z3));
  } else {
    STD_E_ASSERT(mpi_rank == 1);
  // Zone1
    tree z1 = new_UnstructuredZone<I4>("Zone1");

    tree pld11 = new_PointList("PointListDonor",std::vector{101,102,103,104});
    tree gc11 = new_GridConnectivity("Join4","Zone1","CellCenter","Abutting1to1");
    emplace_child(gc11,std::move(pld11));

    tree pld10 = new_PointList("PointListDonor",std::vector{111,112});
    tree gc10 = new_GridConnectivity("Join5","Zone0","Vertex","Abutting1to1");
    emplace_child(gc10,std::move(pld10));

    tree zone_gc1 = new_ZoneGridConnectivity();
    emplace_child(zone_gc1,std::move(gc11));
    emplace_child(zone_gc1,std::move(gc10));
    emplace_child(z1,std::move(zone_gc1));

    emplace_child(b,std::move(z1));

  // Zone2
    tree z2 = new_UnstructuredZone<I4>("Zone2");

    tree pld21 = new_PointList("PointListDonor",std::vector{136,137});
    tree gc21 = new_GridConnectivity("Join6","Zone3","CellCenter","Abutting1to1");
    emplace_child(gc21,std::move(pld21));

    tree zone_gc2 = new_ZoneGridConnectivity();
    emplace_child(zone_gc2,std::move(gc21));
    emplace_child(z2,std::move(zone_gc2));

    emplace_child(b,std::move(z2));
  }
  return b;
}

} // example
