#include "maia/connectivity/std_elements_to_ngons.hpp"
#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "maia/connectivity/iter/connectivity_range.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"
#include "std_e/algorithm/distribution/weighted.hpp"
#include "maia/sids/element_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/future/ranges.hpp"
#include "maia/utils/log/log.hpp"
#include "std_e/logging/mem_logger.hpp" // TODO
#include "maia/sids/maia_cgns.hpp"
#include "maia/transform/renumber/shift.hpp"


using namespace cgns;


namespace maia {


template<class I> auto
concatenate_into_poly_section(const tree_range& sections, ElementType_t elt_type, const std::string& connec_type, I first_id, const std::vector<I>& n_elts, const std::vector<I>& w_elts, MPI_Comm comm) {
  // 0. concatenate all surface elements into one ngon
  int rk = std_e::rank(comm);
  int n_slot = std_e::n_rank(comm);
  int n_section = sections.size();
  auto [dist_cat_elts,dist_cat_connec_elts] = std_e::distribution_weighted_by_blocks(n_slot,n_elts,w_elts);

  std_e::dist_array<I> dconnec(dist_cat_connec_elts,comm); // TODO store a span/ref to the dist in dist_array
  I acc_sz = 0;
  // 0.0. send vertices
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = sections[i];

    auto connec = get_node_value_by_matching<I>(elt_node,connec_type);
    auto connec_dist = get_node_value_by_matching<I8>(elt_node,":CGNS#Distribution/Element");

    std::vector<I> indices(connec.size());
    std::iota(begin(indices),end(indices),acc_sz+connec_dist[0]*w_elts[i]);
    std::vector<I> connec2(connec.begin(),connec.end()); // TODO no copy
    std_e::scatter(dconnec,dist_cat_connec_elts,indices,connec2); // TODO replace by interval-based description of indices
    acc_sz += n_elts[i]*w_elts[i];
  }

  // 0.2. compute displacements (ElementStartOffset)
  //auto [elt_infs,elt_sups] = elements_in_interval(dist_cat_connec_elts[rk],dist_cat_connec_elts[rk+1],n_elts_tot,w_elts);
  std::vector<I> w_elts2(n_section,1);
  auto [elt_infs,elt_sups] = std_e::elements_in_interval(dist_cat_elts[rk],dist_cat_elts[rk+1],n_elts,w_elts2); // TODO simplify since w_elt2==1
  std::vector<I> elt_start_offset(dist_cat_elts[rk+1]-dist_cat_elts[rk] + 1);
  elt_start_offset[0] = dist_cat_connec_elts[rk];

  int k=0;
  for (int i=0; i<n_section; ++i) {
    for (int j=0; j<elt_sups[i]-elt_infs[i]; ++j) {
      elt_start_offset[k+1] = elt_start_offset[k] + w_elts[i];
      ++k;
    }
  }

  // 1. ngon section node
  // connec
  std::vector<I> connec_array(begin(dconnec.local()),end(dconnec.local())); // TODO no copy
  I n_elt_tot = std::accumulate(begin(n_elts),end(n_elts),0);
  tree ngon_section_node = new_Elements(to_string(elt_type),elt_type,std::move(connec_array),first_id,first_id+n_elt_tot-1);

  // elt_start_offset
  tree eso_node = new_DataArray("ElementStartOffset",std::move(elt_start_offset));
  emplace_child(ngon_section_node,std::move(eso_node));

  // distribution
  std::vector<I8> elt_distri_mem(3);
  elt_distri_mem[0] = dist_cat_elts[rk];
  elt_distri_mem[1] = dist_cat_elts[rk+1];
  elt_distri_mem[2] = dist_cat_elts.back();
  tree elt_dist = new_DataArray("Element",std::move(elt_distri_mem));
  std::vector<I8> connec_distri_mem(3);
  connec_distri_mem[0] = dist_cat_connec_elts[rk];
  connec_distri_mem[1] = dist_cat_connec_elts[rk+1];
  connec_distri_mem[2] = dist_cat_connec_elts.back();
  tree connec_dist = new_DataArray("ElementConnectivity",std::move(connec_distri_mem));

  auto dist_node = new_UserDefinedData(":CGNS#Distribution");
  emplace_child(dist_node,std::move(elt_dist));
  emplace_child(dist_node,std::move(connec_dist));
  emplace_child(ngon_section_node,std::move(dist_node));

  return std::make_pair(std::move(dist_cat_elts),std::move(ngon_section_node));
}

auto
replace_faces_by_ngons(tree& z, MPI_Comm comm) -> void {
  // 0. sizes
  auto face_sections = surface_element_sections(z);
  auto cell_sections = volume_element_sections(z);
  int n_face_section = face_sections.size();
  std::vector<I4> n_elts(n_face_section);
  //std::vector<I4> n_elts_tot(n_face_section);
  std::vector<I4> w_elts(n_face_section);
  for (int i=0; i<n_face_section; ++i) {
    const tree& elt_node = face_sections[i];
    auto elt_type = element_type(elt_node);
    auto elt_interval = ElementRange<I4>(elt_node);
    I4 n_elt = elt_interval[1] - elt_interval[0] + 1;
    n_elts[i] = n_elt;
    //n_elts_tot[i] = elt_interval[2];
    w_elts[i] = number_of_vertices(elt_type);
  }

  // 1. shift
  I4 first_3d_elt_id = elements_interval(cell_sections).first();
  I4 n_faces = length(elements_interval(face_sections));
  I4 cell_offset = -first_3d_elt_id + 1 + n_faces; // substract previous start, then start at 1, then leave room for all faces before
  shift_cell_ids(z,cell_offset);

  // 2. concatenate
  // 2.0 connec + eso
  auto [dist_cat_elts,ngon_section_node] = concatenate_into_poly_section(face_sections,cgns::NGON_n,"ElementConnectivity",1,n_elts,w_elts,comm);

  // 2.1 send parents TODO parents_position
  std_e::dist_array<I4> l_parents(dist_cat_elts,comm);
  std_e::dist_array<I4> r_parents(dist_cat_elts,comm);
  I4 acc_sz = 0;
  for (int i=0; i<n_face_section; ++i) {
    const tree& elt_node = face_sections[i];

    auto par = get_node_value_by_matching<I4,2>(elt_node,"ParentElements");
    auto connec_dist = get_node_value_by_matching<I8>(elt_node,":CGNS#Distribution/Element");
    auto dn_elt = connec_dist[1] - connec_dist[0];

    std::vector<I4> indices(dn_elt);
    std::iota(begin(indices),end(indices),acc_sz+connec_dist[0]);

    std::vector<I4> l_par(par.begin()        , par.begin()+dn_elt); // TODO no copy
    std::vector<I4> r_par(par.begin()+dn_elt , par.end()         ); // TODO no copy
    std_e::scatter(l_parents,dist_cat_elts,indices,l_par); // TODO replace by interval-based description of indices
    std_e::scatter(r_parents,dist_cat_elts,indices,r_par); // TODO replace by interval-based description of indices

    acc_sz += n_elts[i];
  }

  // 2.2 parent node
  int rk = std_e::rank(comm);
  auto dn_faces = dist_cat_elts[rk+1] - dist_cat_elts[rk];
  cgns::md_array<I4,2> parents_array(dn_faces,2);
  auto mid_parent_array = std::copy(begin(l_parents.local()),end(l_parents.local()),begin(parents_array));
                          std::copy(begin(r_parents.local()),end(r_parents.local()),mid_parent_array);
  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(ngon_section_node,std::move(parent_elt_node));

  emplace_child(z,std::move(ngon_section_node));
}

auto
replace_elts_by_nfaces(tree& z, MPI_Comm comm) -> void {
  // 0. sizes
  auto cell_sections = volume_element_sections(z);
  int n_cell_section = cell_sections.size();
  std::vector<I4> n_elts(n_cell_section);
  std::vector<I4> w_elts(n_cell_section);
  for (int i=0; i<n_cell_section; ++i) {
    const tree& elt_node = cell_sections[i];
    auto elt_type = element_type(elt_node);
    auto elt_interval = ElementRange<I4>(elt_node);
    I4 n_elt = elt_interval[1] - elt_interval[0] + 1;
    n_elts[i] = n_elt;
    //n_elts_tot[i] = elt_interval[2];
    w_elts[i] = number_of_faces(elt_type);
  }

  // 1. connec + eso
  I4 first_id = element_range(get_child_by_name(z,"NGON_n")).last()+1;
  auto [_1,nfaces_section_node] = concatenate_into_poly_section(cell_sections,cgns::NFACE_n,"CellFace",first_id,n_elts,w_elts,comm);

  emplace_child(z,std::move(nfaces_section_node));
}

auto
turn_info_ngon_nface(tree& z, MPI_Comm comm) -> void {
  auto elt_sections = element_sections(z);
  auto elt_names = elt_sections | std::views::transform([](const tree& x){ return name(x); }) | std_e::to_vector();
  replace_faces_by_ngons(z,comm);
  replace_elts_by_nfaces(z,comm);
  cgns::rm_children_by_names(z,elt_names);
}

auto
std_elements_to_ngons(tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));
  auto _ = maia_perf_log_lvl_0("std_elements_to_ngons");

  generate_interior_faces_and_parents(z,comm);
  turn_info_ngon_nface(z,comm);
}


} // maia
