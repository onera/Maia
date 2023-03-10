#if __cplusplus > 201703L
#include "maia/algo/dist/rearrange_element_sections/rearrange_element_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/interval/interval_sequence.hpp"
#include "std_e/parallel/mpi.hpp"
#include "pdm_multi_block_to_part.h"
#include "std_e/data_structure/multi_range.hpp"
#include "maia/pytree/maia/element_sections.hpp"

using namespace cgns;

namespace maia {

template<class It> auto
merge_same_type_elt_sections(It first_section, It last_section, MPI_Comm comm) -> tree {
  int i_rank = std_e::rank(comm);
  int n_rank = std_e::n_rank(comm);

  int n_section = last_section-first_section;
  STD_E_ASSERT(n_section);
  auto range_first = ElementRange<I4>(*first_section);
  I4 start = range_first[0];
  I4 finish = range_first[1];
  auto elt_type = element_type(*first_section);

  It cur = first_section+1;
  while (cur != last_section) {
    auto cur_range = ElementRange<I4>(*cur);
    I4 cur_start = cur_range[0];
    I4 cur_finish = cur_range[1];
    I4 cur_elt_type = element_type(*first_section);
    STD_E_ASSERT(cur_start == finish+1);
    STD_E_ASSERT(cur_elt_type == elt_type);
    finish = cur_finish;
    ++cur;
  }

  // multi_block_to_part
  std::vector<PDM_g_num_t> multi_distrib_idx(n_section+1);
  multi_distrib_idx[0] = 0;
  std::vector<distribution_vector<PDM_g_num_t>> block_distribs_storer(n_section);
  std::vector<PDM_g_num_t*> block_distribs(n_section);
  for (int i=0; i<n_section; ++i) {
    tree& section_node = *(first_section+i);
    auto section_connec_partial_distri = ElementDistribution<I4>(section_node);
    block_distribs_storer[i] = distribution_from_partial(section_connec_partial_distri,comm);
    block_distribs[i] = block_distribs_storer[i].data();
    multi_distrib_idx[i+1] = multi_distrib_idx[i] + section_connec_partial_distri.back();
  }

  const int n_block = n_section;
  const int n_part = 1;

  std::vector<PDM_g_num_t> merged_distri(n_rank+1);
  std_e::uniform_distribution(begin(merged_distri),end(merged_distri),0,multi_distrib_idx.back());

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
  int stride = cgns::number_of_vertices(elt_type);
  int** stride_one = (int ** ) malloc( n_block * sizeof(int *));
  for(int i_block = 0; i_block < n_block; ++i_block){
    stride_one[i_block] = (int * ) malloc( 1 * sizeof(int));
    stride_one[i_block][0] = stride;
  }

  std::vector<I4*> d_connectivity_sections(n_section);
  for (int i=0; i<n_section; ++i) {
    tree& section_node = *(first_section+i);
    auto section_connec = get_child_value_by_name<I4>(section_node,"ElementConnectivity");
    d_connectivity_sections[i] = section_connec.data();
  }

  I4** parray = nullptr;
  PDM_multi_block_to_part_exch2(mbtp, sizeof(I4), PDM_STRIDE_CST_INTERLACED,
                                stride_one,
                     (void ** ) d_connectivity_sections.data(),
                                nullptr,
                     (void ***) &parray);

  int d_connec_sz = n_elts_0*stride;
  std::vector<I4> d_connectivity_merge(d_connec_sz);
  std::copy_n(parray[0],d_connec_sz,begin(d_connectivity_merge));
  free(parray[0]);
  free(parray);
  PDM_multi_block_to_part_free(mbtp);

  tree elt_node = new_Elements(
    cgns::to_string((ElementType_t)elt_type),
    elt_type,
    std::move(d_connectivity_merge),
    start,finish
  );

  std::vector<I4> partial_dist(3);
  partial_dist[0] = merged_distri[i_rank];
  partial_dist[1] = merged_distri[i_rank+1];
  partial_dist[2] = merged_distri.back();
  tree dist = new_DataArray("Element",std::move(partial_dist));

  tree cgns_dist = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(cgns_dist,std::move(dist));

  emplace_child(elt_node,std::move(cgns_dist));

  return elt_node;
}

auto rearrange_element_sections(tree& b, MPI_Comm comm) -> void {
  auto zs = get_children_by_label(b,"Zone_t");
  if (zs.size()!=1) {
    throw cgns_exception("rearrange_element_sections: only implemented for one zone");
  }
  tree& z = zs[0];

  auto elt_sections = element_sections_ordered_by_range_type_dim(z);

  // Handle NGon case
  if (elt_sections.size()>=1) {
    auto it_ngon = std::find_if(begin(elt_sections),end(elt_sections),[](const tree& n){ return element_type(n)==NGON_n; });
    if (it_ngon != end(elt_sections)) { // found NGon
      if (elt_sections.size()==1) { // only NGon
        return; // only 1 NGon : nothing to merge
      } else {
        if (elt_sections.size()==2) {
          auto it_nface = std::find_if(begin(elt_sections),end(elt_sections),[](const tree& n){ return element_type(n)==NFACE_n; });
          if (it_nface == end(elt_sections)) { // did not found NFace
            throw cgns_exception("Zone "+name(z)+" has a NGon section, but also a section different from NFace, which is forbidden by CGNS/SIDS");
          } else {
            return; // only 1 NGon and 1 NFace : nothing to merge
          }
        } else { // more sections than Ngon+NFace
          throw cgns_exception("Zone "+name(z)+" has a NGon section and a NFace section, but also other element types, which is forbidden by CGNS/SIDS");
        }
      }
    }
  }

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
  // TODO fields
  auto bcs = get_nodes_by_matching(z,"ZoneBC_t/BC_t");
  for (tree& bc : bcs) {
    if (GridLocation(bc)!="Vertex") {
      //auto pl = get_child_value_by_name<I4>(bc,"PointList");
      auto pl = get_child_value_by_name<I4>(bc,"PointList");

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
  auto elt_type = element_type(*section_current);
  auto is_same_elt_type = [&elt_type](const auto& elt_node){ return element_type(elt_node) == elt_type; };
  while (section_current != end(elt_sections)){
    auto section_same_type_end = std::partition_point(section_current,end(elt_sections),is_same_elt_type);
    auto new_section = merge_same_type_elt_sections(section_current,section_same_type_end,comm);
    merged_sections.emplace_back(std::move(new_section));
    section_current = section_same_type_end;
    if (section_current != end(elt_sections)) {
      elt_type = element_type(*section_current);
    }
  }

  rm_children_by_label(z,"Elements_t");
  emplace_children(z,std::move(merged_sections));
}

} // maia
#endif // C++>17
