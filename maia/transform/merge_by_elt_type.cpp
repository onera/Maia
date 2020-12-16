#include "maia/transform/merge_by_elt_type.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "cpp_cgns/sids/sids.hpp"
#include "std_e/algorithm/distribution.hpp"
#include "std_e/interval/knot_sequence.hpp"
#include "std_e/parallel/mpi.hpp"
#include "pdm_multi_block_to_part.h"
#include "std_e/data_structure/multi_range.hpp"


namespace cgns {


template<class Range> auto
distribution_from_partial(const Range& partial_distri, MPI_Comm comm) -> distribution_vector<PDM_g_num_t> {
  int dn_elt = partial_distri[1] - partial_distri[0];
  auto full_distri = distribution_from_dsizes(dn_elt, comm);
  STD_E_ASSERT(full_distri.back()==partial_distri[2]);
  return full_distri;
}

template<class It> auto
merge_same_type_elt_sections(It first_section, It last_section, factory F, MPI_Comm comm) -> tree {
  int i_rank = std_e::rank(comm);
  int n_rank = std_e::nb_ranks(comm);

  int n_section = last_section-first_section;
  STD_E_ASSERT(n_section);
  auto range_first = ElementRange<I4>(*first_section);
  I4 start = range_first[0];
  I4 finish = range_first[1];
  I4 elt_type = ElementType<I4>(*first_section);

  It cur = first_section+1;
  while (cur != last_section) {
    auto cur_range = ElementRange<I4>(*cur);
    I4 cur_start = cur_range[0];
    I4 cur_finish = cur_range[1];
    I4 cur_elt_type = ElementType<I4>(*first_section);
    STD_E_ASSERT(cur_start == finish+1);
    STD_E_ASSERT(cur_elt_type == elt_type);
    finish = cur_finish;
    ++cur;
  }

  // multi_block_to_part
  PDM_MPI_Comm pdm_comm = PDM_MPI_mpi_2_pdm_mpi_comm(&comm);

  std::vector<PDM_g_num_t> multi_distrib_idx(n_section+1);
  multi_distrib_idx[0] = 0;
  std::vector<distribution_vector<PDM_g_num_t>> block_distribs_storer(n_section);
  std::vector<PDM_g_num_t*> block_distribs(n_section);
  for (int i=0; i<n_section; ++i) {
    tree& section_node = *(first_section+i);
    auto section_connec_partial_distri = get_node_value_by_matching<I8>(section_node,":CGNS#Distribution/Distribution");
    block_distribs_storer[i] = distribution_from_partial(section_connec_partial_distri,comm);
    block_distribs[i] = block_distribs_storer[i].data();
    multi_distrib_idx[i+1] = multi_distrib_idx[i] + section_connec_partial_distri.back();
  }

  const int n_block = n_section;
  const int n_part = 1;

  std::vector<PDM_g_num_t> merged_distri(n_rank+1);
  std_e::uniform_distribution(begin(merged_distri),end(merged_distri),1,multi_distrib_idx.back());

  int n_elts_0 = merged_distri[i_rank+1]-merged_distri[i_rank];
  std::vector<PDM_g_num_t> ln_to_gn_0(n_elts_0);
  std::iota(begin(ln_to_gn_0),end(ln_to_gn_0),merged_distri[i_rank]);
  std::vector<PDM_g_num_t*> ln_to_gn = {ln_to_gn_0.data()};
  std::vector<int> n_elts = {n_elts_0};

  PDM_multi_block_to_part_t* mbtp =
    PDM_multi_block_to_part_create(multi_distrib_idx.data(),
                                   n_block,
             (const PDM_g_num_t**) block_distribs.data(),
             (const PDM_g_num_t**) ln_to_gn.data(),
                                   n_elts.data(),
                                   n_part,
                                   pdm_comm);
  int stride = cgns::number_of_nodes(elt_type);
  int** stride_one = (int ** ) malloc( n_block * sizeof(int *));
  for(int i_block = 0; i_block < n_block; ++i_block){
    stride_one[i_block] = (int * ) malloc( 1 * sizeof(int));
    stride_one[i_block][0] = stride;
  }

  std::vector<I4*> d_connectivity_sections(n_section);
  for (int i=0; i<n_section; ++i) {
    tree& section_node = *(first_section+i);
    auto section_connec = get_node_value_by_matching<I4>(section_node,"ElementConnectivity");
    d_connectivity_sections[i] = section_connec.data();
  }

  I4** parray = nullptr;
  PDM_multi_block_to_part_exch2(mbtp, sizeof(I4), PDM_STRIDE_CST,
                                stride_one,
                     (void ** ) d_connectivity_sections.data(),
                                nullptr,
                     (void ***) &parray);

  int d_connec_sz = n_elts_0*stride;
  auto d_connectivity_merge = make_cgns_vector<I4>(d_connec_sz,F.alloc());
  std::copy_n(parray[0],d_connec_sz,begin(d_connectivity_merge));
  free(parray[0]);
  free(parray);
  PDM_multi_block_to_part_free(mbtp);

  tree elt_node = F.newElements(
    cgns::to_string((ElementType_t)elt_type),
    elt_type,
    std_e::make_span(d_connectivity_merge),
    start,finish
  );

  auto partial_dist = make_cgns_vector<I8>(3,F.alloc());
  partial_dist[0] = merged_distri[i_rank];
  partial_dist[1] = merged_distri[i_rank+1];
  partial_dist[2] = merged_distri.back();
  tree dist = F.newDataArray("Distribution",{"I8",{3},partial_dist.data()});

  tree cgns_dist = F.newUserDefinedData(":CGNS#Distribution");
  emplace_child(cgns_dist,std::move(dist));

  emplace_child(elt_node,std::move(cgns_dist));

  return elt_node;
}



auto merge_by_elt_type(tree& b, factory F, MPI_Comm comm) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("merge_by_elt_type: only implemented for one zone");
  }
  tree& z = zs[0];

  auto elt_sections = get_children_by_label(z,"Elements_t");
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_elt_type);

  // 0. new ranges
  I4 new_offset = 1;
  std::vector<I4> old_offsets;
  std::vector<I4> new_offsets;
  for (tree& elt_section : elt_sections) {
    auto range = ElementRange<I4>(elt_section);

    I4 old_offset = range[0];
    old_offsets.push_back(old_offset);

    I4 n_elt = range[1] - range[0] +1;
    range[0] = new_offset;
    range[1] = new_offset + n_elt -1;

    new_offsets.push_back(new_offset);

    new_offset += n_elt;
  }
  old_offsets.push_back(new_offset); // since the range span does not change, last new_offset == last old_offset
  new_offsets.push_back(0); // value unused, but sizes must be equal for zip_sort
  std_e::zip_sort(std::tie(old_offsets,new_offsets));

  // 1. renumber
  // TODO for several zones
  auto bcs = get_nodes_by_matching(z,"ZoneBC_t/BC_t");
  for (tree& bc : bcs) {
    if (to_string(get_child_by_name(bc,"GridLocation").value)!="Vertex") {
      //auto pl = get_child_value_by_name<I4>(bc,"PointList");
      tree& pl_node = get_child_by_name(bc,"PointList");
      auto pl = view_as_span<I4>(pl_node.value);

      for (I4& id : pl) {
        auto index = std_e::interval_index(id,old_offsets);
        auto old_offset = old_offsets[index];
        auto new_offset = new_offsets[index];
        id = id - old_offset + new_offset;
      }
    }
  }

  // 2. merge element sections of the same type
  std::vector<tree> merged_sections;
  auto section_current = begin(elt_sections);
  auto elt_type = ElementType<I4>(*section_current);
  auto is_same_elt_type = [&elt_type](const auto& elt_node){ return ElementType<I4>(elt_node) == elt_type; };
  while (section_current != end(elt_sections)){
    auto section_same_type_end = std::partition_point(section_current,end(elt_sections),is_same_elt_type);
    auto new_section = merge_same_type_elt_sections(section_current,section_same_type_end,F,comm);
    merged_sections.push_back(new_section);
    section_current = section_same_type_end;
    if (section_current != end(elt_sections)) {
      elt_type = ElementType<I4>(*section_current);
    }
  }

  F.rm_children_by_label(z,"Elements_t");
  emplace_children(z,std::move(merged_sections));
}

} // cgns
