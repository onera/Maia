#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"
#include "maia/sids/element_sections.hpp"
#include "maia/sids/maia_cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "std_e/utils/string.hpp"
#include "std_e/future/ranges.hpp"
#include "std_e/algorithm/partition/copy.hpp"
#include "maia/utils/log/log.hpp"
#include "maia/generate/interior_faces_and_parents/scatter_parents_to_boundary_sections.hpp"
#include "maia/generate/interior_faces_and_parents/in_ext_faces_with_parents.hpp"
#include "maia/generate/interior_faces_and_parents/merge_unique_faces.hpp"
//#include "std_e/plog.hpp" // TODO


using namespace cgns;


namespace maia {


template<class I> auto
append_element_section(cgns::tree& z, ElementType_t elt_type, in_faces_with_parents<I>&& faces_with_p, MPI_Comm comm) -> void { // TODO do not copy faces_with_p
  //auto _ = std_e::stdout_time_logger("append_element_section");
  I n_face = faces_with_p.l_parents.size();

  I n_face_tot = std_e::all_reduce(n_face,MPI_SUM,comm);
  I n_face_acc = std_e::scan(n_face,MPI_SUM,comm) - n_face; // TODO exscan

  I last_id = surface_elements_interval(z).last();
  I first_sec_id = last_id+1;
  I last_sec_id = first_sec_id+n_face_tot-1;

  // connec
  tree elt_section_node = new_Elements(to_string(elt_type)+"_interior",(I)elt_type,std::move(faces_with_p.connec),first_sec_id,last_sec_id);

  // parent
  md_array<I,2> parents_array(n_face,2);
  // TODO optim with no copy
  auto mid_parent_array = std::copy(begin(faces_with_p.l_parents),end(faces_with_p.l_parents),begin(parents_array));
                          std::copy(begin(faces_with_p.r_parents),end(faces_with_p.r_parents),mid_parent_array);

  tree parent_elt_node = cgns::new_DataArray("ParentElements",std::move(parents_array));
  emplace_child(elt_section_node,std::move(parent_elt_node));

  // parent position
  md_array<I,2> parent_positions_array(n_face,2);
  // TODO optim with no copy
  auto mid_ppos_array = std::copy(begin(faces_with_p.l_parent_positions),end(faces_with_p.l_parent_positions),begin(parent_positions_array));
                        std::copy(begin(faces_with_p.r_parent_positions),end(faces_with_p.r_parent_positions),mid_ppos_array);

  tree parent_position_elt_node = cgns::new_DataArray("ParentElementsPosition",std::move(parent_positions_array));
  emplace_child(elt_section_node,std::move(parent_position_elt_node));

  // distribution
  std::vector<I8> elt_dist(3);
  elt_dist[0] = n_face_acc;
  elt_dist[1] = n_face_acc + n_face;
  elt_dist[2] = n_face_tot;
  auto dist_node = new_Distribution("Element",std::move(elt_dist));
  emplace_child(elt_section_node,std::move(dist_node));

  emplace_child(z,std::move(elt_section_node));
}


template<class I> auto
fill_cell_face_ids(
  auto& face_in_vol_indices_by_section, auto& face_ids_by_section,
  const std::vector<I>& vol_ids, const std::vector<I>& face_positions, const std::vector<I>& face_ids,
  const auto& vol_section_intervals, const auto& vol_section_types,
  ElementType_t face_type
) {
  // Precondition
  for (I vol_id : vol_ids) {
    STD_E_ASSERT(vol_section_intervals[0]<=vol_id && vol_id<vol_section_intervals.back());
  }

  auto proj = [](I vol_elt_id, auto&&, auto&&){ return vol_elt_id; };
  auto f_copy =
    [face_type,&vol_section_intervals,&vol_section_types,&face_in_vol_indices_by_section,&face_ids_by_section]
      (int index, I vol_id, I face_position, I face_id)
        {
          I vol_index = vol_id-vol_section_intervals[index];
          auto n_face_of_cell = number_of_faces(vol_section_types[index]);
          I first_face_in_vol_index = vol_index*n_face_of_cell;
          I face_in_vol_index = first_face_in_vol_index + (face_position-1); // -1 because CGNS starts at 0
          face_in_vol_indices_by_section[index].push_back(face_in_vol_index);
          face_ids_by_section[index].push_back(face_id);
        };
  std_e::interval_partition_copy(std::tie(vol_ids,face_positions,face_ids),vol_section_intervals,proj,f_copy);
}

template<class I> auto
append_cell_face_info(tree& vol_section, std::vector<I> face_in_vol_indices, std::vector<I> face_ids, MPI_Comm comm) {
  auto partial_dist = ElementDistribution(vol_section);
  auto dist_cell_face = maia::distribution_from_partial(partial_dist,comm);
  std_e::scale(dist_cell_face,number_of_faces(element_type(vol_section)));
  std_e::dist_array<I> cell_face(dist_cell_face,comm);
  //auto _ = maia_perf_log_lvl_2("scatter cell_face");
  std_e::scatter(cell_face,dist_cell_face,std::move(face_in_vol_indices),face_ids);
  //_.stop();

  std::vector<I> connec_array(begin(cell_face.local()),end(cell_face.local()));
  emplace_child(vol_section,cgns::new_DataArray("CellFace",std::move(connec_array)));
}

template<class I> auto
fill_cell_face_info(
  const tree_range& vol_sections,
  const in_ext_faces_with_parents<I>& faces,
  ElementType_t face_type,
  I first_ext_id, I first_in_id,
  auto& face_in_vol_indices_by_section, auto& face_ids_by_section,
  MPI_Comm comm
)
{
  //auto _ = std_e::stdout_time_logger("fill_cell_face_info");
  auto vol_section_intervals = element_sections_interval_vector<I>(vol_sections);
  auto vol_section_types = vol_sections
                         | std::views::transform([](const tree& e){ return element_type(e); })
                         | std_e::to_vector();

  const auto& ext_face_ids  = faces.ext.boundary_parents;
  const auto& ext_parents   = faces.ext.vol_parents;
  const auto& ext_face_pos  = faces.ext.vol_parent_positions;


  I n_face = faces.in.size();
  I n_face_acc = std_e::scan(n_face,MPI_SUM,comm) - n_face; // TODO exscan
  const auto& in_face_ids  = std_e::iota(n_face,first_in_id+n_face_acc);
  const auto& in_l_parents  = faces.in .l_parents;
  const auto& in_l_face_pos = faces.in .l_parent_positions;
  const auto& in_r_parents  = faces.in .r_parents;
  const auto& in_r_face_pos = faces.in .r_parent_positions;

  fill_cell_face_ids(face_in_vol_indices_by_section,face_ids_by_section,  ext_parents ,ext_face_pos ,ext_face_ids,  vol_section_intervals, vol_section_types,face_type);
  fill_cell_face_ids(face_in_vol_indices_by_section,face_ids_by_section,  in_l_parents,in_l_face_pos,in_face_ids ,  vol_section_intervals, vol_section_types,face_type);
  fill_cell_face_ids(face_in_vol_indices_by_section,face_ids_by_section,  in_r_parents,in_r_face_pos,in_face_ids ,  vol_section_intervals, vol_section_types,face_type);

}

template<class I> auto
_generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));

  // -1. base queries
  auto elt_sections = element_sections(z);
  auto vol_sections = volume_element_sections(z);

  // 0. Generate faces
  auto faces_and_parents_sections = generate_element_faces_and_parents<I>(elt_sections);

  // 1. Make them unique
  I n_vtx = cgns::VertexSize_U<I>(z);
  I first_3d_elt_id = volume_elements_interval(z).first();
  auto unique_faces_by_section = merge_unique_faces(faces_and_parents_sections,n_vtx,first_3d_elt_id,comm);
  auto& unique_faces_tri  = unique_faces_by_section[0];
  auto& unique_faces_quad = unique_faces_by_section[1];

  // 3. Compute cell_face
  /// ElementRange ext
  auto tri_ext_section = unique_element_section(z,TRI_3);
  auto quad_ext_section = unique_element_section(z,QUAD_4);
  I first_tri_ext_id = element_range(tri_ext_section).first();
  I first_quad_ext_id = element_range(quad_ext_section).first();
  /// ElementRange in
  I first_tri_in_id = first_3d_elt_id; // because first_3d_elt_id is the first volume element and tri_in will take its place
  I first_quad_in_id = first_tri_in_id + std_e::all_reduce(unique_faces_tri.in.size(),MPI_SUM,comm);

  // cell face info
  int n_cell_section = vol_sections.size();
  std::vector<std::vector<I>> face_in_vol_indices_by_section(n_cell_section);
  std::vector<std::vector<I>> face_ids_by_section(n_cell_section);

  fill_cell_face_info(vol_sections,unique_faces_tri ,TRI_3 ,first_tri_ext_id ,first_tri_in_id , face_in_vol_indices_by_section,face_ids_by_section,comm);
  fill_cell_face_info(vol_sections,unique_faces_quad,QUAD_4,first_quad_ext_id,first_quad_in_id, face_in_vol_indices_by_section,face_ids_by_section,comm);

  // 2. Scatter
  // 2.0 parent_element info back to original exterior faces
  scatter_parents_to_boundary_sections(elt_sections,unique_faces_tri .ext,cgns::TRI_3 ,comm);
  scatter_parents_to_boundary_sections(elt_sections,unique_faces_quad.ext,cgns::QUAD_4,comm);
  // 2.1 cell face info
  for (int i=0; i<n_cell_section; ++i) {
    append_cell_face_info(vol_sections[i],std::move(face_in_vol_indices_by_section[i]),std::move(face_ids_by_section[i]),comm);
  }

  // 4. Create new interior faces sections
  append_element_section(z,cgns::TRI_3 ,std::move(unique_faces_tri .in),comm);
  append_element_section(z,cgns::QUAD_4,std::move(unique_faces_quad.in),comm);
};

auto
generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));
  auto _ = maia_perf_log_lvl_1("generate_interior_faces_and_parents");
  if (value(z).data_type()=="I4") return _generate_interior_faces_and_parents<I4>(z,comm);
  if (value(z).data_type()=="I8") return _generate_interior_faces_and_parents<I8>(z,comm);
  throw cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}


} // maia
