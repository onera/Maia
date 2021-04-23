#include "maia/transform/fsdm_distribution.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "cpp_cgns/tree_manip.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "maia/sids/element_sections.hpp"
#include "pdm_multi_block_to_part.h"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/buffer/buffer_vector.hpp"
#include <algorithm>

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
  for (tree& z : get_children_by_label(b,"Zone_t")) {
    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    for (tree& bc : cgns::get_nodes_by_matching(z,"ZoneBC/BC_t")) {
      auto pl = cgns::PointList<I4>(bc);
      auto fields = std::vector<std::vector<double>>{}; // TODO extract these fields (if they exist)
      auto [new_pl,_] = redistribute_to_match_face_dist(elt_dists,elt_intervals,pl,fields,comm);

      rm_child_by_name(bc,"PointList");

      std_e::buffer_vector<I4> pl_buf(begin(new_pl),end(new_pl));
      cgns::emplace_child(bc,new_PointList("PointList",std::move(pl_buf)));
    }
    // TODO update BC distribution
  }
}


auto
distribute_vol_fields_to_match_global_element_range(cgns::tree& b, MPI_Comm comm) -> void {
  STD_E_ASSERT(b.label=="CGNSBase_t");
  int i_rank = std_e::rank(comm);
  int n_rank = std_e::nb_ranks(comm);

  for (tree& z : get_children_by_label(b,"Zone_t")) {
    int n_cell = cgns::CellSize_U<I4>(z);

    auto elt_sections = element_sections_ordered_by_range(z);
    auto elt_intervals = elt_interval_range<I4>(elt_sections);
    auto elt_dists = elt_distributions(elt_sections,comm);

    int n_section = elt_sections.size();

    auto section_is_2d = [](const tree& x){ return element_dimension(element_type(x))==2; };
    STD_E_ASSERT(std::is_partitioned(begin(elt_sections),end(elt_sections),section_is_2d));
    auto first_section_3d = std::partition_point(begin(elt_sections),end(elt_sections),section_is_2d);
    int n_2d_section = first_section_3d - begin(elt_sections);
    auto elt_3d_sections = std_e::make_span(elt_sections.data()+n_2d_section,elt_sections.data()+n_section);
    auto elt_3d_intervals = std_e::make_span(elt_intervals.data()+n_2d_section,elt_intervals.data()+n_section+1);
    auto elt_3d_dists = std_e::make_span(elt_dists.data()+n_2d_section,elt_dists.data()+n_section+1);

    int n_3d_section = n_section-n_2d_section;

    // multi-block to part
    std::vector<PDM_g_num_t> multi_distrib_idx(elt_3d_intervals.begin(),elt_3d_intervals.end());
    std_e::offset(multi_distrib_idx,-elt_3d_intervals[0]);
    std::vector<distribution_vector<PDM_g_num_t>> block_distribs_storer(n_3d_section);
    std::vector<PDM_g_num_t*> block_distribs(n_3d_section);
    std::vector<int> d_elt_szs(n_3d_section);
    for (int i=0; i<n_3d_section; ++i) {
      tree& section_node = elt_3d_sections[i];
      auto section_connec_partial_distri = get_node_value_by_matching<I8>(section_node,":CGNS#Distribution/Element");
      d_elt_szs[i] = section_connec_partial_distri[1]-section_connec_partial_distri[0];
      block_distribs_storer[i] = distribution_from_partial(section_connec_partial_distri,comm);
      block_distribs[i] = block_distribs_storer[i].data();
    }

    const int n_block = n_3d_section;
    const int n_part = 1;

    std::vector<PDM_g_num_t> merged_distri(n_rank+1);
    std_e::uniform_distribution(begin(merged_distri),end(merged_distri),0,n_cell);

    int n_elts_0 = merged_distri[i_rank+1]-merged_distri[i_rank];
    std::vector<PDM_g_num_t> ln_to_gn_0(n_elts_0);
    std::iota(begin(ln_to_gn_0),end(ln_to_gn_0),merged_distri[i_rank]+1);
    std::vector<PDM_g_num_t*> ln_to_gn = {ln_to_gn_0.data()};
    std::vector<int> n_elts = {n_elts_0};

    PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&comm);
    PDM_multi_block_to_part_t* mbtp =
      PDM_multi_block_to_part_create(multi_distrib_idx.data(),
                                     n_block,
               (const PDM_g_num_t**) block_distribs.data(),
               (const PDM_g_num_t**) ln_to_gn.data(),
                                     n_elts.data(),
                                     n_part,
                                     pdm_comm);
    int stride = 1;
    int** stride_one = (int**) malloc( n_block * sizeof(int*));
    for(int i_block = 0; i_block < n_block; ++i_block){
      stride_one[i_block] = (int * ) malloc( 1 * sizeof(int));
      stride_one[i_block][0] = stride;
    }

    tree_range flow_sol_nodes = cgns::get_children_by_labels(z,{"FlowSolution_t","DiscreteData_t"});
    for (tree& flow_sol_node : flow_sol_nodes) {
      tree_range sol_nodes = cgns::get_children_by_label(flow_sol_node,"DataArray_t");
      for (tree& sol_node : sol_nodes) {
        auto sol = cgns::view_as_span<R8>(sol_node.value);
        std::vector<R8*> darray_ptr(n_block);
        int offset = 0;
        for (int i=0; i<n_block; ++i) {
          darray_ptr[i] = sol.data()+offset;
          offset += d_elt_szs[i];
        }

        R8** parray = nullptr;
        PDM_multi_block_to_part_exch2(mbtp, sizeof(R8), PDM_STRIDE_CST,
                                      stride_one,
                           (void ** ) darray_ptr.data(),
                                      nullptr,
                           (void ***) &parray);

        int d_sol_sz = merged_distri[i_rank+1] - merged_distri[i_rank];
        std_e::buffer_vector<R8> new_sol(d_sol_sz);
        std::copy_n(parray[0],d_sol_sz,begin(new_sol));
        free(parray[0]);
        free(parray);

        sol_node.value = cgns::make_node_value(std::move(new_sol));
      }
    }
    PDM_multi_block_to_part_free(mbtp);

    tree& z_dist_node = cgns::get_child_by_name(z,":CGNS#Distribution");
    std_e::buffer_vector<I8> cell_partial_dist = {merged_distri[i_rank],merged_distri[i_rank+1],merged_distri.back()};
    emplace_child(z_dist_node,new_DataArray("Cell",std::move(cell_partial_dist)));
  }
}


} // maia
