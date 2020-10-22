#include "maia/transform/__old/renumber_point_lists.hpp"

#include "std_e/future/contract.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/Building_Block_Structure_Definitions.hpp"
#include <iostream> // TODO


namespace cgns {


auto
renumber_point_list(std_e::span<I4> pl, const std_e::offset_permutation<I4>& permutation) -> void {
  // Precondition: permutation is an index permutation (i.e. sort(permutation) == integer_range(permutation.size()))
  std_e::apply(permutation,pl);
}
auto
renumber_point_list2(std_e::span<I4> pl, const std_e::offset_permutation<I4>& permutation) -> void {
  // Precondition: permutation is an index permutation (i.e. sort(permutation) == integer_range(permutation.size()))
  for (auto& i : pl) {
    //STD_E_ASSERT(i>0 && i<=permutation.perm.size());
    i = permutation(i);
    if (i==0) i=-1; // TODO FIXME this is because offset_permutation also offsets -1 rather than not messing with this special value
    //STD_E_ASSERT(i>0 && i<=permutation.perm.size());
  }
}

template<class Fun> auto
for_each_point_list(tree& z, const std::string& grid_location, Fun f) {
  STD_E_ASSERT(z.label=="Zone_t");
  std::vector<std::string> search_gen_paths = {"ZoneBC/BC_t","ZoneGridConnectivity/GridConnectivity_t"};
  for (tree& bc : get_nodes_by_matching(z,search_gen_paths)) {
    if (GridLocation(bc)==grid_location) {
      f(get_child_by_name(bc,"PointList").value);
    }
  }
}

auto
is_bc_gc_with_empty_point_list(const tree& t) -> bool {
  return 
      (t.label=="BC_t" || t.label=="GridConnectivity_t")
   && get_child_by_name(t,"PointList").value.dims[1]==0;
}
auto
remove_if_empty_point_list(tree& z, factory F) -> void {
  STD_E_ASSERT(z.label=="Zone_t");
  // TODO this is ugly, think of something better
  std::vector<std::string> search_node_names = {"ZoneBC","ZoneGridConnectivity"};
  for (const auto& search_node_name : search_node_names) {
    if (has_child_of_name(z,search_node_name)) {
      tree& search_node = get_child_by_name(z,search_node_name);
      F.rm_children_by_predicate(search_node, is_bc_gc_with_empty_point_list);
    }
  }
}

auto
renumber_point_lists(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void {
  auto f = [&permutation](auto& pl){ renumber_point_list(view_as_span<I4>(pl),permutation); };
  for_each_point_list(z,grid_location,f);
}
auto
renumber_point_lists2(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void {
  auto f = [&permutation](auto& pl){ renumber_point_list2(view_as_span<I4>(pl),permutation); };
  for_each_point_list(z,grid_location,f);
}

auto
renumber_point_lists_donated(donated_point_lists& plds, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void {
  // TODO replace by multi_range iteration
  for (auto& pld : plds) {
    if (to_string(pld.loc)==grid_location) {
      renumber_point_list2(pld.pl,permutation);
    }
  }
}


auto
rm_invalid_ids_in_point_list(node_value& pl, factory F) -> void {
  auto old_pl_val = view_as_span<I4>(pl);
  auto new_pl_val = make_cgns_vector<I4>(F.alloc());
  std::copy_if(begin(old_pl_val),end(old_pl_val),std::back_inserter(new_pl_val),[](I4 i){ return i!=-1; });
  F.deallocate_node_value(pl);
  pl = view_as_node_value_1(new_pl_val);
}
auto
rm_invalid_ids_in_point_lists(tree& z, const std::string& grid_location, factory F) -> void {
  auto f = [&F](auto& pl){ rm_invalid_ids_in_point_list(pl,F); };
  for_each_point_list(z,grid_location,f);
  remove_if_empty_point_list(z,F);
}
auto
rm_invalid_ids_in_point_lists_with_donors(tree& z, const std::string& grid_location, factory F) -> void {
  STD_E_ASSERT(z.label=="Zone_t");
  for (tree& bc : get_nodes_by_matching(z,"ZoneGridConnectivity/GridConnectivity_t")) {
    if (GridLocation(bc)==grid_location) {
      node_value& pl = get_child_by_name(bc,"PointList").value;
      node_value& pld = get_child_by_name(bc,"PointListDonor").value;
      auto old_pl_val  = view_as_span<I4>(pl);
      auto old_pld_val = view_as_span<I4>(pld);
      auto new_pl_val  = make_cgns_vector<I4>(F.alloc());
      auto new_pld_val = make_cgns_vector<I4>(F.alloc());
      int old_nb_pl = old_pl_val.size();
      for (int i=0; i<old_nb_pl; ++i) {
        //STD_E_ASSERT(old_pld_val[i]!=-1); // if donor, then it means that it was owned by the donor zone, hence, not deleted by it
        if (old_pl_val[i]!=-1 && old_pld_val[i]!=-1) {
          new_pl_val.push_back(old_pl_val[i]);
          new_pld_val.push_back(old_pld_val[i]);
        }
      }
      F.deallocate_node_value(pl);
      F.deallocate_node_value(pld);
      pl = view_as_node_value_1(new_pl_val);
      pld = view_as_node_value_1(new_pld_val);
    }
  }
  remove_if_empty_point_list(z,F);
}

auto
rm_grid_connectivities(tree& z, const std::string& grid_location, factory F) -> void {
  STD_E_ASSERT(z.label=="Zone_t");
  tree& zgc = get_child_by_name(z,"ZoneGridConnectivity");
  F.rm_children_by_predicate(zgc, [&](const tree& n){ return n.label=="GridConnectivity_t" && GridLocation(n)==grid_location; });
}


} // cgns
