#pragma once


#include <vector>
#include "cpp_cgns/cgns.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/struct/in_ext_faces_by_section.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"
#include "cpp_cgns/sids/creation.hpp"


namespace maia {


template<class I> auto
scatter_parents_to_boundary(cgns::tree& bnd_section, const ext_faces_with_parents<I>& ext_faces, MPI_Comm comm) {
  // 1. Boundary face indices in the array are just boundary face parent element ids, but starting at 0
  auto first_face_id = ElementRange<I>(bnd_section)[0];
  std::vector<I> face_indices = ext_faces.bnd_face_parent_elements;
  std_e::offset(face_indices,-first_face_id);

  // 2. Send parent cell info to the boundary faces
  // 2.0. Protocol creation
  auto partial_distri = ElementDistribution<I>(bnd_section);
  auto distri = maia::distribution_from_partial(partial_distri,comm);
  auto sp = create_exchange_protocol(distri,std::move(face_indices));

  // 2.1. parent_elements
  std_e::dist_array<I> pe(distri,comm);
  std_e::scatter(sp,ext_faces.cell_parent_elements,pe);

  I n_face_res = pe.local().size();
  cgns::md_array<I,2> pe_array(n_face_res,2);
  std::ranges::copy(pe.local(),begin(pe_array)); // only half is assign, the other is 0

  cgns::tree pe_node = cgns::new_DataArray("ParentElements",std::move(pe_array));
  emplace_child(bnd_section,std::move(pe_node));

  // 2.2. parent_positions
  std_e::dist_array<I> pp(distri,comm);
  std_e::scatter(sp,ext_faces.cell_parent_positions,pp);

  cgns::md_array<I,2> pp_array(n_face_res,2);
  std::ranges::copy(pp.local(),begin(pp_array)); // only half is assign, the other is 0

  cgns::tree pp_node = cgns::new_DataArray("ParentElementsPosition",std::move(pp_array));
  emplace_child(bnd_section,std::move(pp_node));
}


template<class Tree_range, class I> auto
scatter_parents_to_boundary_sections(Tree_range& bnd_sections, const in_ext_faces_by_section<I>& unique_faces_sections, MPI_Comm comm) -> void {
  for(const auto& fs : unique_faces_sections) {
    auto bnd_section = std::ranges::find_if(bnd_sections,[&fs](const cgns::tree& t){ return element_type(t)==fs.face_type ; });
    if (bnd_section != end(bnd_sections)) {
      scatter_parents_to_boundary(*bnd_section,fs.ext,comm);
    } else {
      STD_E_ASSERT(fs.ext.size()==0);
    }
  }
}


} // maia
