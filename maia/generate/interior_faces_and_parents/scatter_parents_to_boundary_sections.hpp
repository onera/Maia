#pragma once


#include <vector>
#include "cpp_cgns/cgns.hpp"
#include "maia/generate/interior_faces_and_parents/struct/in_ext_faces_by_section.hpp"
#include "maia/utils/parallel/utils.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"


namespace maia {


// TODO review
template<class I> auto
scatter_parents_to_boundary(cgns::tree& bnd_section, const ext_faces_with_parents<I>& ext_faces, MPI_Comm comm) {
  //auto _ = std_e::stdout_time_logger("scatter_parents_to_boundary");
  auto partial_dist = ElementDistribution(bnd_section);
  auto dist_I8 = maia::distribution_from_partial(partial_dist,comm);
  auto first_face_id = cgns::get_node_value_by_matching<I>(bnd_section,"ElementRange")[0];
  std::vector<I> face_indices = ext_faces.boundary_parents;
  std_e::offset(face_indices,-first_face_id);

  // parents
  std_e::dist_array<I> parents(dist_I8,comm);
  std_e::scatter(parents,dist_I8,face_indices,ext_faces.vol_parents);

  I n_face_res = parents.local().size();
  cgns::md_array<I,2> parents_array(n_face_res,2);
  std::copy(begin(parents.local()),end(parents.local()),begin(parents_array)); // only half is assign, the other is 0

  cgns::tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(bnd_section,std::move(parent_elt_node));

  // parent positions
  std_e::dist_array<I> parent_positions(dist_I8,comm);
  std_e::scatter(parent_positions,dist_I8,std::move(face_indices),ext_faces.vol_parent_positions); // TODO protocol (here, we needlessly recompute with face_indices)

  cgns::md_array<I,2> parent_positions_array(n_face_res,2);
  std::copy(begin(parent_positions.local()),end(parent_positions.local()),begin(parent_positions_array)); // only half is assign, the other is 0

  cgns::tree parent_position_elt_node = cgns::new_DataArray("ParentElementsPosition",std::move(parent_positions_array));
  emplace_child(bnd_section,std::move(parent_position_elt_node));
}


template<class I, class Tree_range> auto
scatter_parents_to_boundary_sections(
  Tree_range& elt_sections,
  cgns::ElementType_t face_type,
  const ext_faces_with_parents<I>& ext_faces,
  MPI_Comm comm
)
 -> void
{
  auto bnd_section = std::ranges::find_if(elt_sections,[face_type](const cgns::tree& t){ return element_type(t)==face_type ; });
  if (bnd_section == end(elt_sections)) {
    STD_E_ASSERT(ext_faces.size()==0);
  } else {
    scatter_parents_to_boundary(*bnd_section,ext_faces,comm);
  }
}


template<class Tree_range, class I> auto
scatter_parents_to_boundary_sections(Tree_range& elt_sections, const in_ext_faces_by_section<I>& unique_faces_sections, MPI_Comm comm) -> void {
  for(const auto& fs : unique_faces_sections) {
    scatter_parents_to_boundary_sections(elt_sections,fs.face_type,fs.ext,comm);
  }
}


} // maia
