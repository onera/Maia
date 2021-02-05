#pragma once


#include "cpp_cgns/cgns.hpp"
#include "std_e/future/span.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/neighbor_graph.hpp"
#include "maia/transform/__old/donated_point_lists.hpp"

#include "range/v3/view/group_by.hpp"
#include "range/v3/algorithm/sort.hpp"
#include "range/v3/action/sort.hpp"
#include "range/v3/view/zip.hpp"


namespace cgns {


struct interzone_point_list_info {
  zone_exchange ze;
  pl_by_donor_zone pld_by_z;
};

// TODO also return the GridLocation
inline auto find_point_list_by_zone_donor(pl_by_donor_zone& pl_by_z, const std::string& z_name) -> donated_point_lists {
  // TODO use std::find_if (need iterators in jagged_range)
  donated_point_lists res;
  int nb_pl_donors = pl_by_z.donor_z_names.size();
  for (int i=0; i<nb_pl_donors; ++i) {
    if (pl_by_z.donor_z_names[i] == z_name) {
      res.push_back({
        pl_by_z.receiver_z_names[i],
        pl_by_z.locs[i],
        pl_by_z.plds[i]
      });
    }
  }
  return res;
}


// declarations {
template<class Fun> auto apply_base_renumbering(tree& b, factory F, Fun zone_renumbering, MPI_Comm comm) -> void; // TODO no default
auto register_connectivities_PointList_infos(tree& base, MPI_Comm comm) -> interzone_point_list_info;
auto re_number_point_lists_donors(interzone_point_list_info& pl_infos) -> void;
// declarations }


template<class Fun> auto
apply_base_renumbering(tree& b, factory F, Fun zone_renumbering, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");

  interzone_point_list_info pl_infos;
  //if (std_e::nb_ranks(comm)>1) { // TODO clean (does not work for sequential with several GC)
    pl_infos = register_connectivities_PointList_infos(b,comm);
  //}

  for (tree& z : zs) {
    auto z_plds = find_point_list_by_zone_donor(pl_infos.pld_by_z,z.name);
    zone_renumbering(z,z_plds,F);
  }

  //if (std_e::nb_ranks(comm)>1) { // TODO clean
    re_number_point_lists_donors(pl_infos);
  //}
}

inline auto
symmetrize_grid_connectivities(tree& b, factory F, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");

  auto zs = get_children_by_label(b,"Zone_t");

  zone_exchange ze(b,comm);
  // note: pl[i] and plds[i] belong to the same GC (they are matched together) // TODO zip everything
  auto pls_by_zone = ze.send_PointList_to_donor_proc();
  auto plds_by_zone = ze.send_PointListDonor_to_donor_proc();

  for (tree& z : zs) {
    tree& zgc = cgns::get_child_by_name(z,"ZoneGridConnectivity");
    ELOG(z.name);
    auto z_pls  = find_point_list_by_zone_donor(pls_by_zone ,z.name) | ranges::actions::sort(less_receiver_zone);
    auto z_plds = find_point_list_by_zone_donor(plds_by_zone,z.name) | ranges::actions::sort(less_receiver_zone);
    auto pls_by_recv_z  = z_pls  | ranges::views::group_by(eq_receiver_zone);
    auto plds_by_recv_z = z_plds | ranges::views::group_by(eq_receiver_zone);
    auto gc_by_recv_z = ranges::views::zip(pls_by_recv_z,plds_by_recv_z);
    auto z_gcs = cgns::get_nodes_by_matching(zgc,"GridConnectivity_t");
    for (const auto& gcs : gc_by_recv_z) {
      std::string receiver_z_name = gcs.first[0].receiver_z_name;
      auto pl = make_cgns_vector<I4>(F.alloc());
      for (const auto& recv_pld : gcs.second) { // Note : using pl DONOR of the OPPOSITE zone, because they are the pl of the CURRENT zone
        for (I4 i : recv_pld.pl) {
          pl.push_back(i);
        }
      }

      auto pld = make_cgns_vector<I4>(F.alloc());
      for (const auto& recv_pl : gcs.first) { // Note : inverted for the same reason
        for (I4 i : recv_pl.pl) {
          pld.push_back(i);
        }
      }

      for (tree& z_gc : z_gcs) {
        auto opp_z_name = to_string(z_gc.value);
        if (opp_z_name == receiver_z_name) {
          ELOG(opp_z_name);
          auto z_pl = view_as_span<I4>(get_child_by_name(z_gc,"PointList").value);
          for (I4 i : z_pl) {
            pl.push_back(i);
          }
          auto z_pld = view_as_span<I4>(get_child_by_name(z_gc,"PointListDonor").value);
          for (I4 i : z_pld) {
            pld.push_back(i);
          }
        }
      }
      // TODO sort unique

      tree pl_node  = F.new_PointList("PointList"     ,std_e::make_span(pl ));
      tree pld_node = F.new_PointList("PointListDonor",std_e::make_span(pld));
      tree new_gc =  F.new_GridConnectivity("Sym_GC_"+receiver_z_name,receiver_z_name,"Vertex","Abutting1to1"); // TODO Vertex
      emplace_child(new_gc,std::move(pl_node));
      emplace_child(new_gc,std::move(pld_node));
      emplace_child(zgc,std::move(new_gc));
    }
  }

}



} // cgns
