#if __cplusplus > 201703L
#include "maia/algo/dist/fsdm_distribution/fsdm_distribution.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/pytree/maia/element_sections.hpp"
#include "pdm_multi_block_to_part.h"
#include "std_e/algorithm/algorithm.hpp"
#include <algorithm>

using namespace cgns;

namespace maia {

// TODO I4 -> I
auto add_fsdm_distribution(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(label(b)=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("add_fsdm_distribution (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  I8 n_vtx = VertexSize_U<I8>(z);
  I8 n_vtx_owned = n_vtx = get_node_value_by_matching<I8>(z,":CGNS#LocalNumbering/VertexSizeOwned")[0];;
  auto vtx_distri = distribution_from_dsizes(n_vtx_owned, comm);
  auto partial_vtx_distri = full_to_partial_distribution(vtx_distri,comm);
  std::vector<I4> vtx_distri_mem(begin(partial_vtx_distri),end(partial_vtx_distri));
  tree vtx_dist = new_DataArray("Vertex",std::move(vtx_distri_mem));
  auto dist_node = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(dist_node,std::move(vtx_dist));
  emplace_child(z,std::move(dist_node));

  auto elt_sections = get_children_by_label(z,"Elements_t");
  for (tree& elt_section : elt_sections) {
    auto elt_range = ElementRange<I4>(elt_section);
    I4 n_owned_elt = elt_range[1] - elt_range[0] + 1;

    auto elt_distri = distribution_from_dsizes(n_owned_elt, comm);
    auto partial_elt_distri = full_to_partial_distribution(elt_distri,comm);
    tree elt_dist = new_DataArray("Element",std::move(partial_elt_distri));

    auto dist_node = new_UserDefinedData(":CGNS#Distribution");
    emplace_child(dist_node,std::move(elt_dist));
    emplace_child(elt_section,std::move(dist_node));
  }
}

template<class I, class Tree_range> auto
elt_interval_range(const Tree_range& sorted_elt_sections) {
  int n_elt = sorted_elt_sections.size();
  std::vector<I> interval_rng(n_elt+1);

  for (int i=0; i<n_elt; ++i) {
    interval_rng[i] = ElementRange<I>(sorted_elt_sections[i])[0];
  }

  interval_rng[n_elt] = ElementRange<I>(sorted_elt_sections.back())[1]+1; // +1 because CGNS intervals are closed, we want open

  return interval_rng;
}

template<class Tree_range> auto
elt_distributions(const Tree_range& sorted_elt_sections, MPI_Comm comm) {
  int n_elt = sorted_elt_sections.size();
  std::vector<distribution_vector<I4>> dists(n_elt);
  for (int i=0; i<n_elt; ++i) {
    const tree& elt = sorted_elt_sections[i];
    auto partial_dist = ElementDistribution<PDM_g_num_t>(elt);
    auto dist = distribution_from_partial(partial_dist,comm);
    dists[i] = distribution_vector<I4>(dist.n_interval()); // TODO make resize accessible
    std::copy(begin(dist),end(dist),begin(dists[i]));
  }
  return dists;
}


auto
distribute_bc_ids_to_match_face_dist(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(label(b)=="CGNSBase_t");
  for (tree& z : get_children_by_label(b,"Zone_t")) {
    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    for (tree& bc : cgns::get_nodes_by_matching(z,"ZoneBC/BC_t")) {
      auto pl = cgns::PointList<I4>(bc);

      auto field_nodes = cgns::get_nodes_by_matching(bc,"BCDataSet_t/BCData_t/DataArray_t");
      int n_fields = field_nodes.size();
      std::vector<std_e::span<R8>> fields(n_fields);
      for (int i=0; i<n_fields; ++i) {
        fields[i] = cgns::view_as_span<R8>(value(field_nodes[i]));
      }

      auto [new_dist,new_pl,new_fields] = redistribute_to_match_face_dist(elt_dists,elt_intervals,pl,fields,comm);

      std::vector<I8> dims = {1,(I8)new_pl.size()}; // required by SIDS (9.3: BC_t)
      node_value new_pl_value(std::move(new_pl),std::move(dims));
      value(cgns::get_node_by_name(bc,"PointList")) = std::move(new_pl_value);
      value(cgns::get_node_by_name(bc,":CGNS#Distribution")) = std::move(new_dist);

      for (int i=0; i<n_fields; ++i) {
        value(field_nodes[i]) = std::move(new_fields[i]);
      }
    }
  }
}


} // maia
#endif // C++>17
