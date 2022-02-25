#include "maia/generate/interior_faces_and_parents/append_interior_faces_sections.hpp"
#include "cpp_cgns/sids/creation.hpp"

#include "std_e/parallel/mpi/collective/scan.hpp"
#include "std_e/parallel/mpi/collective/reduce.hpp"
#include "maia/sids/element_sections.hpp"


using namespace cgns;


namespace maia {


template<class I> auto
create_interior_faces_section(in_faces_with_parents<I>&& fps, I section_first_id, ElementType_t face_type, MPI_Comm comm) {
  I n_face = fps.size();

  I n_face_tot = std_e::all_reduce(n_face,MPI_SUM,comm);
  I n_face_acc = std_e::ex_scan(n_face,MPI_SUM,0,comm);

  // connectivity
  I section_last_id = section_first_id+n_face_tot-1;
  tree elt_section_node = new_Elements(to_string(face_type)+"_interior",face_type,std::move(fps.connec),section_first_id,section_last_id);

  // parent element
  md_array<I,2> pe_array(n_face,2);
  auto [_0,mid_pe_array] = std::ranges::copy(fps.l_parent_elements,begin(pe_array));
                           std::ranges::copy(fps.r_parent_elements,mid_pe_array);

  tree pe_node = cgns::new_DataArray("ParentElements",std::move(pe_array));
  emplace_child(elt_section_node,std::move(pe_node));

  // parent position
  md_array<I,2> pp_array(n_face,2);
  auto [_1,mid_pp_array] = std::ranges::copy(fps.l_parent_positions,begin(pp_array));
                           std::ranges::copy(fps.r_parent_positions,mid_pp_array);

  tree parent_position_elt_node = cgns::new_DataArray("ParentElementsPosition",std::move(pp_array));
  emplace_child(elt_section_node,std::move(parent_position_elt_node));

  // distribution
  std::vector<I8> elt_dist(3);
  elt_dist[0] = n_face_acc;
  elt_dist[1] = n_face_acc + n_face;
  elt_dist[2] = n_face_tot;
  auto dist_node = new_Distribution("Element",std::move(elt_dist));
  emplace_child(elt_section_node,std::move(dist_node));

  return std::make_pair(n_face_tot,std::move(elt_section_node));
}


template<class I> auto
append_interior_faces_sections(cgns::tree& z, in_ext_faces_by_section<I>&& faces_sections, I first_interior_face_id, MPI_Comm comm) -> I {
  I current_section_id = first_interior_face_id;
  for(auto& fs : faces_sections) {
    auto [n_in_faces,in_section_node] = create_interior_faces_section(std::move(fs.in),current_section_id,fs.face_type,comm);
    current_section_id += n_in_faces;
    emplace_child(z,std::move(in_section_node));
  }
  return current_section_id;
}


// Explicit instanciations of functions defined in this .cpp file
template auto append_interior_faces_sections(cgns::tree& z, in_ext_faces_by_section<I4>&& faces_sections, I4 first_interior_face_id, MPI_Comm comm) -> I4;
template auto append_interior_faces_sections(cgns::tree& z, in_ext_faces_by_section<I8>&& faces_sections, I8 first_interior_face_id, MPI_Comm comm) -> I8;


} // maia
