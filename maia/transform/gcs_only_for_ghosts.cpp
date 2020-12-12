#include "maia/transform/gcs_only_for_ghosts.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "std_e/log.hpp"
#include <algorithm>


namespace cgns {


auto gcs_only_for_ghosts(tree& b, factory F) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  for (tree& z : zs) {
    int n_node = VertexSize_U<I4>(z);
    auto n_ghost_node = get_node_value_by_matching<I4>(z,"GridCoordinates/FSDM#n_ghost_node")[0];
    int n_owned_node = n_node - n_ghost_node;
    auto gcs = get_nodes_by_matching(z,"ZoneGridConnectivity_t/GridConnectivity_t");
    for (tree& gc : gcs) {
      if (GridLocation(gc)=="Vertex") {
        tree& pl_node  = get_child_by_name(gc,"PointList");
        tree& pld_node = get_child_by_name(gc,"PointListDonor");

        auto old_pl  = view_as_span<I4>(pl_node .value);
        auto old_pld = view_as_span<I4>(pld_node.value);

        I4 n_old_id = old_pl.size();

        auto new_pl  = make_cgns_vector<I4>(F.alloc());
        auto new_pld = make_cgns_vector<I4>(F.alloc());

        for (int i=0; i<n_old_id; ++i) {
          if (old_pl[i]>n_owned_node)  {
            new_pl .push_back(old_pl [i]);
            new_pld.push_back(old_pld[i]);
          }
        }

        F.deallocate_node_value(pl_node.value);
        pl_node.value = view_as_node_value(new_pl);
        pl_node.value.dims = {1,new_pl.size()}; // required by SIDS

        F.deallocate_node_value(pld_node.value);
        pld_node.value = view_as_node_value(new_pld);
        pld_node.value.dims = {1,new_pld.size()}; // required by SIDS
      }
    }
  }
}


} // cgns
