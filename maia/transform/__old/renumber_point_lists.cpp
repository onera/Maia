#include "maia/transform/__old/renumber_point_lists.hpp"

#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/future/contract.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "cpp_cgns/sids/Building_Block_Structure_Definitions.hpp"
#include <iostream> // TODO


using cgns::tree;
using cgns::node_value;
using cgns::I4;
using cgns::I8;


namespace maia {


template<class I> auto
renumber_point_list(std_e::span<I> pl, const std_e::offset_permutation<I>& permutation) -> void {
  // Precondition: permutation is an index permutation (i.e. sort(permutation) == integer_range(permutation.size()))
  std_e::apply(permutation,pl);
}
template<class I> auto
renumber_point_lists(const std::vector<std_e::span<I>>& pls, const std_e::offset_permutation<I>& permutation) -> void {
  for (auto pl : pls) {
    renumber_point_list(pl,permutation);
  }
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
  STD_E_ASSERT(label(z)=="Zone_t");
  std::vector<std::string> search_gen_paths = {"ZoneBC/BC_t","ZoneGridConnectivity/GridConnectivity_t"}; // TODO and other places!
  for (tree& bc : get_nodes_by_matching(z,search_gen_paths)) {
    if (GridLocation(bc)==grid_location) {
      f(value(get_child_by_name(bc,"PointList")));
    }
  }
}

auto
is_bc_gc_with_empty_point_list(const tree& t) -> bool {
  return
      (label(t)=="BC_t" || label(t)=="GridConnectivity_t")
   && value(get_child_by_name(t,"PointList")).extent(1)==0;
}
auto
remove_if_empty_point_list(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  // TODO this is ugly, think of something better
  std::vector<std::string> search_node_names = {"ZoneBC","ZoneGridConnectivity"};
  for (const auto& search_node_name : search_node_names) {
    if (has_child_of_name(z,search_node_name)) {
      tree& search_node = get_child_by_name(z,search_node_name);
      rm_children_by_predicate(search_node, is_bc_gc_with_empty_point_list);
    }
  }
}

template<class I> auto
renumber_point_lists(tree& z, const std_e::offset_permutation<I>& permutation, const std::string& grid_location) -> void {
  auto f = [&permutation](auto& pl){ renumber_point_list(cgns::view_as_span<I>(pl),permutation); };
  for_each_point_list(z,grid_location,f);
}
auto
renumber_point_lists2(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void {
  auto f = [&permutation](auto& pl){ renumber_point_list2(cgns::view_as_span<I4>(pl),permutation); };
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
rm_invalid_ids_in_point_list(node_value& pl) -> void {
  auto old_pl_val = view_as_span<I4>(pl);
  std::vector<I4> new_pl_val;
  std::copy_if(begin(old_pl_val),end(old_pl_val),std::back_inserter(new_pl_val),[](I4 i){ return i!=-1; });
  pl = node_value(std::move(new_pl_val));
}
auto
rm_invalid_ids_in_point_lists(tree& z, const std::string& grid_location) -> void {
  for_each_point_list(z,grid_location,rm_invalid_ids_in_point_list);
  remove_if_empty_point_list(z);
}
auto
rm_invalid_ids_in_point_lists_with_donors(tree& z, const std::string& grid_location) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  for (tree& bc : get_nodes_by_matching(z,"ZoneGridConnectivity/GridConnectivity_t")) {
    if (GridLocation(bc)==grid_location) {
      node_value& pl = value(get_child_by_name(bc,"PointList"));
      node_value& pld = value(get_child_by_name(bc,"PointListDonor"));
      auto old_pl_val  = view_as_span<I4>(pl);
      auto old_pld_val = view_as_span<I4>(pld);
      std::vector<I4> new_pl_val;
      std::vector<I4> new_pld_val;
      int old_nb_pl = old_pl_val.size();
      for (int i=0; i<old_nb_pl; ++i) {
        //STD_E_ASSERT(old_pld_val[i]!=-1); // if donor, then it means that it was owned by the donor zone, hence, not deleted by it
        if (old_pl_val[i]!=-1 && old_pld_val[i]!=-1) {
          new_pl_val.push_back(old_pl_val[i]);
          new_pld_val.push_back(old_pld_val[i]);
        }
      }
      pl = node_value(std::move(new_pl_val));
      pld = node_value(std::move(new_pld_val));
    }
  }
  remove_if_empty_point_list(z);
}

auto
rm_grid_connectivities(tree& z, const std::string& grid_location) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  tree& zgc = get_child_by_name(z,"ZoneGridConnectivity");
  rm_children_by_predicate(zgc, [&](const tree& n){ return label(n)=="GridConnectivity_t" && GridLocation(n)==grid_location; });
}


template auto renumber_point_lists<I4>(tree& z, const std_e::offset_permutation<I4>& permutation, const std::string& grid_location) -> void;
template auto renumber_point_lists<I8>(tree& z, const std_e::offset_permutation<I8>& permutation, const std::string& grid_location) -> void;
template auto renumber_point_lists<I4>(const std::vector<std_e::span<I4>>& pls, const std_e::offset_permutation<I4>& permutation) -> void;
template auto renumber_point_lists<I8>(const std::vector<std_e::span<I8>>& pls, const std_e::offset_permutation<I8>& permutation) -> void;

} // maia
