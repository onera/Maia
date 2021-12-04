#include "maia/connectivity/std_elements_to_ngons.hpp"
#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "maia/connectivity/iter/connectivity_range.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"
#include "maia/sids/element_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/logging/time_logger.hpp" // TODO


using namespace cgns;


namespace maia {

auto replace_faces_by_ngons(tree& z, MPI_Comm comm) -> void {
  auto _ = std_e::stdout_time_logger("== replace_faces_by_ngons ==");
  int rk = std_e::rank(comm);
  // 0. sizes
  auto surf_elt_sections = surface_element_sections(z);
  I4 n_surf_elt_tot = 0;
  int n_section = surf_elt_sections.size();
  std::vector<I4> n_elts(n_section);
  //std::vector<I4> n_elts_tot(n_section);
  std::vector<I4> w_elts(n_section);
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = surf_elt_sections[i];
    auto elt_type = element_type(elt_node);
    auto elt_interval = ElementRange<I4>(elt_node);
    I4 n_elt = elt_interval[1] - elt_interval[0] + 1;
    n_elts[i] = n_elt;
    //n_elts_tot[i] = elt_interval[2];
    w_elts[i] = number_of_nodes(elt_type);
    n_surf_elt_tot += n_elt;
  }
  // 1. shift volume element ids
  I4 first_3d_elt_id = volume_elements_interval(z).first();
  //I4 volume_elts_offset = n_surf_elt_tot+1 - first_3d_elt_id;
  I4 volume_elts_offset = 1 - first_3d_elt_id; // TODO here this is done to please cassiopee, but not CGNS compliant
  for (tree& elt_node : surf_elt_sections) {
    auto pe = get_node_value_by_matching<I4,2>(elt_node,"ParentElements");
    for (I4& id : pe) {
      if (id != 0) {
        id += volume_elts_offset;
      }
    }
  }

  // 2. concatenate all surface elements into one ngon
  int n_slot = std_e::n_rank(comm);
  auto [dist_cat_elts,dist_cat_connec_elts] = std_e::distribution_weighted_by_blocks(n_slot,n_elts,w_elts);

  std_e::dist_array<I4> dconnec(dist_cat_connec_elts,comm); // TODO store a span/ref to the dist in dist_array
  I4 acc_sz = 0;
  // 2.0. send vertices
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = surf_elt_sections[i];

    auto connec = get_node_value_by_matching<I4>(elt_node,"ElementConnectivity");
    auto connec_dist = get_node_value_by_matching<I8>(elt_node,":CGNS#Distribution/Element");

    std::vector<I4> indices(connec.size());
    std::iota(begin(indices),end(indices),acc_sz+connec_dist[0]*w_elts[i]);
    std::vector<I4> connec2(connec.begin(),connec.end()); // TODO no copy
    std_e::scatter(dconnec,dist_cat_connec_elts,indices,connec2); // TODO replace by interval-based description of indices
    acc_sz += n_elts[i]*w_elts[i];
  }

  //// 2.1. send parents OLD not working
  //auto dist_cat_par = dist_cat_elts;
  //std_e::scale(dist_cat_par,2); // TODO ugly
  //std_e::dist_array<I4> parents(dist_cat_par,comm); // store a span/ref to the dist in dist_array
  //acc_sz = 0;
  //for (int i=0; i<n_section; ++i) {
  //  const tree& elt_node = surf_elt_sections[i];

  //  auto par = get_node_value_by_matching<I4,2>(elt_node,"ParentElements");
  //  auto connec_dist = get_node_value_by_matching<I8>(elt_node,":CGNS#Distribution/Element");

  //  std::vector<I4> indices(par.size());
  //  std::iota(begin(indices),end(indices),acc_sz+connec_dist[0]*2);
  //  std::vector<I4> par2(par.begin(),par.end()); // TODO no copy
  //  std_e::scatter(parents,dist_cat_par,indices,par2); // TODO replace by interval-based description of indices
  //  acc_sz += n_elts[i]*2;
  //}
  // 2.1. send parents OTHER SOL
  std_e::dist_array<I4> l_parents(dist_cat_elts,comm);
  std_e::dist_array<I4> r_parents(dist_cat_elts,comm);
  acc_sz = 0;
  for (int i=0; i<n_section; ++i) {
    const tree& elt_node = surf_elt_sections[i];

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
  // 2.1. compute displacements (ElementStartOffset)
  //auto [elt_infs,elt_sups] = elements_in_interval(dist_cat_connec_elts[rk],dist_cat_connec_elts[rk+1],n_elts_tot,w_elts);
  std::vector<I4> w_elts2(n_section,1);
  auto [elt_infs,elt_sups] = std_e::elements_in_interval(dist_cat_elts[rk],dist_cat_elts[rk+1],n_elts,w_elts2); // TODO simplify since w_elt2==1
  std::vector<I4> elt_start_offset(dist_cat_elts[rk+1]-dist_cat_elts[rk] + 1);
  elt_start_offset[0] = dist_cat_connec_elts[rk];

  //int acc = 0;
  //for (int i=0; i<n_section; ++i) { // TODO DEL
  //  acc += elt_sups[i]-elt_infs[i];
  //}
  //STD_E_ASSERT(acc==dist_cat_connec_elts[rk]);

  int k=0;
  for (int i=0; i<n_section; ++i) {
    for (int j=0; j<elt_sups[i]-elt_infs[i]; ++j) {
      elt_start_offset[k+1] = elt_start_offset[k] + w_elts[i];
      ++k;
    }
  }

  // 3. ngon section node
  // connec
  std::vector<I4> connec_array(begin(dconnec.local()),end(dconnec.local()));
  tree ngon_section_node = new_Elements("NGons",(I4)cgns::NGON_n,std::move(connec_array),1,n_surf_elt_tot);

  // elt_start_offset
  tree eso_node = new_DataArray("ElementStartOffset",std::move(elt_start_offset));
  emplace_child(ngon_section_node,std::move(eso_node));

  // parent
  auto dn_faces = dist_cat_elts[rk+1] - dist_cat_elts[rk];
  cgns::md_array<I4,2> parents_array(dn_faces,2);
  auto mid_parent_array = std::copy(begin(l_parents.local()),end(l_parents.local()),begin(parents_array));
                          std::copy(begin(r_parents.local()),end(r_parents.local()),mid_parent_array);
  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(ngon_section_node,std::move(parent_elt_node));

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

  cgns::rm_children_by_label(z,"Elements_t");
  emplace_child(z,std::move(ngon_section_node));
}

auto std_elements_to_ngons(tree& z, MPI_Comm comm) -> void {
  generate_interior_faces_and_parents(z,comm);
  replace_faces_by_ngons(z,comm);
}


} // maia
