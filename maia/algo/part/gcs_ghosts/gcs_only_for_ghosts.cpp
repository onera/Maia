#if __cplusplus > 201703L
#include "maia/algo/part/gcs_ghosts/gcs_only_for_ghosts.hpp"
#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids.hpp"
#include <algorithm>
#include "maia/factory/partitioning/gc_name_convention.hpp"


namespace cgns {


auto gcs_only_for_ghosts(tree& b) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  for (tree& z : zs) {
    auto n_vtx_owned = get_node_value_by_matching<I4>(z,":CGNS#LocalNumbering/VertexSizeOwned")[0];

    auto gcs = get_nodes_by_matching(z,"ZoneGridConnectivity_t/GridConnectivity_t");
    for (tree& gc : gcs) {
      if (GridLocation(gc)=="Vertex") {
        tree& pl_node  = get_child_by_name(gc,"PointList");
        tree& pld_node = get_child_by_name(gc,"PointListDonor");

        auto old_pl  = get_value<I4>(pl_node );
        auto old_pld = get_value<I4>(pld_node);

        I4 n_old_id = old_pl.size();

        std::vector<I4> new_pl;
        std::vector<I4> new_pld;

        for (int i=0; i<n_old_id; ++i) {
          bool owned = (old_pl[i] <= n_vtx_owned);
          if (!owned) {
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
