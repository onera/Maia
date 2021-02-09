#include "maia/transform/gcs_only_for_ghosts.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include <algorithm>
#include "maia/partitioning/gc_name_convention.hpp"
#include "std_e/buffer/buffer_vector.hpp"


namespace cgns {


auto gcs_only_for_ghosts(tree& b) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  for (tree& z : zs) {
    int n_node = VertexSize_U<I4>(z);
    auto ghost_info = get_node_value_by_matching<I4>(z,":CGNS#Ppart/np_vtx_ghost_information");
    // n_ghost
    int zone_proc = maia::proc_of_zone(z.name);
    auto is_owned = [=](I4 ghost_proc){ return ghost_proc==-1 || ghost_proc==zone_proc; };
    auto first_ghost = std::partition_point(begin(ghost_info),end(ghost_info),is_owned);
    I4 n_ghost = end(ghost_info)-first_ghost;
    tree& grid_coord_node = get_child_by_name(z,"GridCoordinates");
    cgns::emplace_child(grid_coord_node,new_UserDefinedData("FSDM#n_ghost",n_ghost));
    // GCs
    auto gcs = get_nodes_by_matching(z,"ZoneGridConnectivity_t/GridConnectivity_t");
    for (tree& gc : gcs) {
      if (GridLocation(gc)=="Vertex") {
        int opp_zone_proc = maia::opp_proc_of_grid_connectivity(gc.name);
        tree& pl_node  = get_child_by_name(gc,"PointList");
        tree& pld_node = get_child_by_name(gc,"PointListDonor");

        auto old_pl  = view_as_span<I4>(pl_node .value);
        auto old_pld = view_as_span<I4>(pld_node.value);

        I4 n_old_id = old_pl.size();

        std_e::buffer_vector<I4> new_pl;
        std_e::buffer_vector<I4> new_pld;

        for (int i=0; i<n_old_id; ++i) {
          I4 owner_proc = ghost_info[old_pl[i]-1]; // -1 because PointList (refering to a vertex) is 1-indexed in CGNS
          if (owner_proc!=zone_proc && owner_proc==opp_zone_proc)  {
            new_pl .push_back(old_pl [i]);
            new_pld.push_back(old_pld[i]);
          }
        }

        pl_node.value = make_node_value(std::move(new_pl));
        pl_node.value.dims = {1,pl_node.value.dims[0]}; // required by SIDS

        pld_node.value = make_node_value(std::move(new_pld));
        pld_node.value.dims = {1,pld_node.value.dims[0]}; // required by SIDS
      }
    }
  }
}


} // cgns
