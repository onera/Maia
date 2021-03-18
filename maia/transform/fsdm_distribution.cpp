#include "maia/transform/fsdm_distribution.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "std_e/buffer/buffer_vector.hpp"
#include "maia/transform/utils.hpp"

using namespace cgns;

namespace maia {

auto add_fsdm_distribution(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("add_fsdm_distribution (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  int n_rank = std_e::nb_ranks(comm);

  auto n_vtx = VertexSize_U<I4>(z);
  auto n_ghost_node = 0;
  if (cgns::has_node(z,"GridCoordinates/FSDM#n_ghost")) {
    n_ghost_node = get_node_value_by_matching<I4>(z,"GridCoordinates/FSDM#n_ghost")[0];
  }
  I4 n_owned_vtx = n_vtx - n_ghost_node;
  auto vtx_distri = distribution_from_dsizes(n_owned_vtx, comm);
  auto partial_vtx_distri = full_to_partial_distribution(vtx_distri,comm);
  std_e::buffer_vector<I8> vtx_distri_mem(begin(partial_vtx_distri),end(partial_vtx_distri));
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
    std_e::buffer_vector<I8> elt_distri_mem(begin(partial_elt_distri),end(partial_elt_distri));

    I4 elt_type = ElementType<I4>(elt_section);
    tree elt_dist = new_DataArray("Element",std::move(elt_distri_mem));

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
    auto partial_dist = cgns::get_node_value_by_matching<I8>(elt,":CGNS#Distribution/Element");
    dists[i] = distribution_from_partial(partial_dist,comm);
  }
  return dists;
}


auto
distribute_bc_ids_to_match_face_dist(tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("distribute_bc_ids_to_match_face_dist (as FSDM) expects only one zone per process");
  }
  tree& z = zs[0];

  auto elt_sections = element_sections_ordered_by_range(z);
  auto elt_intervals = elt_interval_range<I4>(elt_sections);
  auto elt_dists = elt_distributions(elt_sections,comm);

  for (tree& bc : cgns::get_nodes_by_matching(z,"ZoneBC/BC_t")) {
    auto pl = cgns::PointList<I4>(bc);
    auto fields = std::vector<std::vector<double>>{}; // TODO extract these fields (if they exist)
    ELOG(bc.name);
    auto [new_pl,_] = redistribute_to_match_face_dist(elt_dists,elt_intervals,pl,fields,comm);

    rm_child_by_name(bc,"PointList");

    std_e::buffer_vector<I4> pl_buf(begin(new_pl),end(new_pl));
    cgns::emplace_child(bc,new_PointList("PointList",std::move(pl_buf)));
  }
  // TODO update BC distribution
}

} // maia
