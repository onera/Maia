#if __cplusplus > 201703L
#include "maia/__old/transform/remove_ghost_info.hpp"

#include "maia/__old/transform/base_renumbering.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/Building_Block_Structure_Definitions.hpp"
#include "std_e/algorithm/iota.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "maia/__old/transform/renumber_point_lists.hpp"

using cgns::tree;
using cgns::I4;
using cgns::I8;
using namespace cgns; // TODO

namespace maia {

auto
remove_ghost_info(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(label(b)=="CGNSBase_t");
  apply_base_renumbering(b,remove_ghost_info_from_zone,comm);

  for (tree& z : get_children_by_label(b,"Zone_t")) {
    rm_invalid_ids_in_point_lists(z,"FaceCenter"); // For BC
    rm_invalid_ids_in_point_lists_with_donors(z,"Vertex"); // For GC
    rm_grid_connectivities(z,"FaceCenter");
    rm_grid_connectivities(z,"CellCenter");
  }

  symmetrize_grid_connectivities(b,comm);
}


template<class I> auto
nb_elements(const tree& elt_pool) -> I {
  auto elt_range = ElementRange<I4>(elt_pool);
  return elt_range[1]-elt_range[0]+1; // CGNS ranges are closed
}
template<class I> auto
nb_ghost_elements(const tree& elt_pool) -> I {
  if (has_child_of_name(elt_pool,"Rind")) {
    auto elt_rind = Rind<I4>(elt_pool);
    STD_E_ASSERT(elt_rind[0]==0);
    return elt_rind[1];
  } else {
    return 0;
  }
}
template<class I> auto
nb_owned_elements(const tree& elt_pool) -> I {
  return nb_elements<I>(elt_pool) - nb_ghost_elements<I>(elt_pool);
}

// TODO BC, renum connec vertices after
/**
 * \brief remove all ghost info, but keeps the GridCOnnectivity 1-to-1 on nodes
 * \param [inout] z: the cgns zone
 * \param [in] plds: PointListDonors from other zones pointing to zone "z"
 * \pre ghost nodes/elements are given by "Rind" nodes
 * \pre ghost nodes/elements are given their "owner" zone and id by GridConnectivities
 *
 * \details
 *   1. remove ghost elements (rind + grid connectivity)
 *   2. renumber elts in point lists
 *   3. remove ghost nodes (rind)
 *   4. only keep those used in conncectivities
 *   5. renumber nodes in GridCoordinates, PointList, PointListDonor from other zones, Elements
 *   6. delete invalid PointList
*/
auto
remove_ghost_info_from_zone(tree& z, donated_point_lists& plds) -> void {
  tree_range elt_pools = get_children_by_label(z,"Elements_t");
  std::sort(begin(elt_pools),end(elt_pools),compare_by_range);
  STD_E_ASSERT(cgns::elts_ranges_are_contiguous(elt_pools));
  int nb_elt_pools = elt_pools.size();

  // 0. compute elt permutation
  I4 elt_first_id = ElementRange<I4>(elt_pools[0])[0];
  std::vector<I4> permutation;
  std_e::interval_vector<I4> intervals = {0};
  I4 nb_owned_cells = 0;
  for (const tree& elt_pool: elt_pools) {
    ElementType_t elt_type = element_type(elt_pool);
    I4 nb_owned_elts = nb_owned_elements<I4>(elt_pool);
    I4 nb_ghost_elts = nb_ghost_elements<I4>(elt_pool);

    if (element_dimension(elt_type)==3) {
      nb_owned_cells += nb_owned_elts;
    }

    std_e::iota_n(std::back_inserter(permutation), nb_owned_elts, intervals.back());
    intervals.push_back_length(nb_owned_elts);
    std::fill_n(std::back_inserter(permutation), nb_ghost_elts, -1-1); // UGLY: -offset (here: -1) because just after, offset_permutation does +offset
  }
  std_e::offset_permutation perm(elt_first_id,1,permutation);

  // 1. renum pl BC
  renumber_point_lists(z,perm,"FaceCenter");
  //renumber_point_lists(z,perm,"CellCenter");
  //renumber_point_lists_donated(plds,perm,"FaceCenter");
  //renumber_point_lists_donated(plds,perm,"CellCenter");

  // 2. rm ghost cells
  CellSize_U<I4>(z) = nb_owned_cells;
  for (int i=0; i<nb_elt_pools; ++i) {
    tree& elt_pool = elt_pools[i];
    auto elt_range = ElementRange<I4>(elt_pool);
    elt_range[0] = intervals[i]+1;
    elt_range[1] = intervals[i+1];

    auto elt_type = element_type(elt_pool);
    tree& elt_connec = get_child_by_name(elt_pool,"ElementConnectivity");
    // TODO once pybind11: do not allocate/copy/del, only resize
    //elt_connec.value.dims[0] = intervals.length(i)*number_of_vertices(elt_type);
    // del old {
    int new_connec_size = intervals.length(i)*number_of_vertices(elt_type);
    auto old_connec_val = get_value<I4>(elt_connec);
    std::vector<I4> new_connec_val(new_connec_size);
    for (int i=0; i<new_connec_size; ++i) {
      new_connec_val[i] = old_connec_val[i];
    }
    value(elt_connec) = node_value(std::move(new_connec_val));
    // del old }

    rm_child_by_name(elt_pool,"Rind");
  }


  // 4. gather all nodes used by elements
  /// 4.1. counting impl
  int old_nb_nodes = VertexSize_U<I4>(z);
  std::vector<I4> nodes2(old_nb_nodes,-1);
  for (const tree& elt_pool: elt_pools) {
    auto cs = ElementConnectivity<I4>(elt_pool);
    for (I4 c : cs) {
      nodes2[c-1] = 0;
    }
  }
  // TODO remplate by filter(!=-1) | iota
  I4 cnt = 0;
  for (I4& n : nodes2) {
    if (n!=-1) {
      n = cnt++;
    }
  }
  // 4.1. }

  int nb_nodes2 = cnt;

  // 5. delete unused nodes
  VertexSize_U<I4>(z) = nb_nodes2;

  tree& coords = get_child_by_name(z,"GridCoordinates");
  rm_child_by_name(coords,"Rind");

  /// 5.0. renumber GridCoordinates
  for (tree& coord : get_children_by_label(coords,"DataArray_t")) {
    // TODO once pybind11: do not allocate/copy/del, only resize
    auto old_coord_val = get_value<R8>(coord);
    std::vector<R8> new_coord_val(nb_nodes2);
    for (I4 i=0; i<old_nb_nodes; ++i) {
      I4 new_node_pos = nodes2[i];
      if (new_node_pos!=-1) {
        new_coord_val[new_node_pos] = old_coord_val[i];
      }
    }
    value(coord) = node_value(std::move(new_coord_val));
  }

  //auto node_perm_at_0 = node_perm.perm;
  //auto old_to_new_node_perm = std_e::inverse_partial_permutation(node_perm.perm,old_nb_nodes,-1);
  //auto old_to_new_node_perm = std_e::inverse_partial_permutation(node_perm_at_0,old_nb_nodes,-1);
  /// 5.1. renumber element connectivities
  for (tree& elt_pool: elt_pools) {
    auto connec = get_child_value_by_name<I4>(elt_pool,"ElementConnectivity");
    for (I4& node : connec) {
      node = nodes2[node-1]+1; // TODO invert in offset_permutation
    }
  }

  /// 5.2. renumber pl
  std_e::offset_permutation node_perm2(1,std::move(nodes2)); // CGNS nodes indexed at 1
  renumber_point_lists2(z,node_perm2,"Vertex");
  renumber_point_lists_donated(plds,node_perm2,"Vertex");
}

} // maia
#endif // C++>17
