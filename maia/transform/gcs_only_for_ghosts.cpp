#if __cplusplus > 201703L
#include "maia/transform/gcs_only_for_ghosts.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids.hpp"
#include <algorithm>
#include "maia/partitioning/gc_name_convention.hpp"


namespace cgns {


auto gcs_only_for_ghosts(tree& b) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  for (tree& z : zs) {
    auto ghost_info = get_node_value_by_matching<I4>(z,":CGNS#Ppart/np_vtx_ghost_information");
    // n_ghost
    int zone_proc = maia::proc_of_zone(name(z));
    auto is_owned = [=](I4 ghost_proc){ return ghost_proc==-1 || ghost_proc==zone_proc; };
    auto first_ghost = std::partition_point(begin(ghost_info),end(ghost_info),is_owned);
    I4 n_ghost = end(ghost_info)-first_ghost;
    tree& grid_coord_node = get_child_by_name(z,"GridCoordinates");
    cgns::emplace_child(grid_coord_node,new_UserDefinedData("FSDM#n_ghost",n_ghost));
    // GCs
    auto gcs = get_nodes_by_matching(z,"ZoneGridConnectivity_t/GridConnectivity_t");
    for (tree& gc : gcs) {
      if (GridLocation(gc)=="Vertex") {
        int opp_zone_proc = maia::opp_proc_of_grid_connectivity(name(gc));
        tree& pl_node  = get_child_by_name(gc,"PointList");
        tree& pld_node = get_child_by_name(gc,"PointListDonor");

        auto old_pl  = get_value<I4>(pl_node );
        auto old_pld = get_value<I4>(pld_node);

        I4 n_old_id = old_pl.size();

        std::vector<I4> new_pl;
        std::vector<I4> new_pld;

        for (int i=0; i<n_old_id; ++i) {
          I4 owner_proc = ghost_info[old_pl[i]-1]; // -1 because PointList (refering to a vertex) is 1-indexed in CGNS
          if (owner_proc!=zone_proc && owner_proc==opp_zone_proc)  {
            new_pl .push_back(old_pl [i]);
            new_pld.push_back(old_pld[i]);
          }
        }

        std::vector<I8> pl_dims = {1,(I8)new_pl.size()};
        value(pl_node) = node_value(std::move(new_pl),pl_dims);

        std::vector<I8> pld_dims = {1,(I8)new_pld.size()};
        value(pld_node) = node_value(std::move(new_pld),pld_dims);
      }
    }
  }
}


} // cgns
#endif // C++>17
