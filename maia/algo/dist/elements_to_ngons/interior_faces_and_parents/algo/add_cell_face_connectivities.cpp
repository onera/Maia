#if __cplusplus > 201703L
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/algo/add_cell_face_connectivities.hpp"

#include "cpp_cgns/sids/creation.hpp"
#include "std_e/parallel/struct/distributed_array.hpp"
#include "std_e/algorithm/partition/copy.hpp"
#include "std_e/future/ranges.hpp"
#include "std_e/parallel/mpi/collective/scan.hpp"
#include "std_e/parallel/mpi/collective/reduce.hpp"
#include "std_e/algorithm/algorithm.hpp"
#include "maia/pytree/maia/element_sections.hpp"
#include "maia/utils/parallel/distribution.hpp"
#include "std_e/data_structure/multi_range/multi_range.hpp"


using namespace cgns;


namespace maia {


template<class I>
struct cell_face_info {
  std::vector<I> cell_indices;
  std::vector<I> cell_face_ids;
};

template<class I> auto
add_face_info(cell_face_info<I>& x, const auto& face_info, ElementType_t cell_type, I cell_first_id) -> void {
  auto [cell_id,face_position_in_cell,face_id] = face_info;
  I cell_index = cell_id-cell_first_id;
  auto n_face_of_cell = number_of_faces(cell_type);
  I first_face_in_vol_index = cell_index*n_face_of_cell;
  I face_in_vol_index = first_face_in_vol_index + (face_position_in_cell-1); // -1 because CGNS starts at 1
  x.cell_indices .push_back(face_in_vol_index);
  x.cell_face_ids.push_back(face_id);
}

template<class I> auto
add_cell_face_infos(
  std::vector<cell_face_info<I>>& cell_face_info_by_section,
  const auto& faces_info,
  const auto& cell_section_intervals, const auto& cell_section_types
)
 -> void
{
  // Precondition
  for (auto [cell_id,_0,_1] : faces_info) {
    STD_E_ASSERT(cell_section_intervals[0]<=cell_id && cell_id<cell_section_intervals.back());
  }

  // Send each face_info to its parent cell section
  auto proj = [](const auto& face_info){
    auto [cell_id,_0,_1] = face_info;
    return cell_id;
  };
  auto f_copy = [&](int index, const auto& face_info) {
    add_face_info(cell_face_info_by_section[index], face_info, cell_section_types[index], cell_section_intervals[index]);
  };
  std_e::interval_partition_copy(faces_info,cell_section_intervals,proj,f_copy);
}

template<class I> auto
fill_cell_face_info(
  std::vector<cell_face_info<I>>& cell_face_info_by_section,
  const tree_range& cell_sections,
  const in_ext_faces<I>& faces,
  I first_in_id,
  MPI_Comm comm
)
{
  auto cell_section_intervals = element_sections_interval_vector<I>(cell_sections);
  auto cell_section_types = cell_sections
                          | std::views::transform([](const tree& e){ return element_type(e); })
                          | std_e::to_vector();

  const auto& ext_face_ids = faces.ext.bnd_face_parent_elements;
  const auto& ext_pe       = faces.ext.cell_parent_elements;
  const auto& ext_pp       = faces.ext.cell_parent_positions;

  I n_face = faces.in.size();
  I n_face_acc = std_e::ex_scan(n_face,MPI_SUM,0,comm);
  const auto& in_face_ids = std_e::iota_vector(n_face,first_in_id+n_face_acc);
  const auto& in_l_pe = faces.in.l_parent_elements;
  const auto& in_l_pp = faces.in.l_parent_positions;
  const auto& in_r_pe = faces.in.r_parent_elements;
  const auto& in_r_pp = faces.in.r_parent_positions;

  auto  ext_faces_info = std_e::view_as_multi_range( ext_pe,  ext_pp, ext_face_ids);
  auto in_l_faces_info = std_e::view_as_multi_range(in_l_pe, in_l_pp, in_face_ids);
  auto in_r_faces_info = std_e::view_as_multi_range(in_r_pe, in_r_pp, in_face_ids);

  add_cell_face_infos(cell_face_info_by_section,   ext_faces_info,  cell_section_intervals, cell_section_types);
  add_cell_face_infos(cell_face_info_by_section,  in_l_faces_info,  cell_section_intervals, cell_section_types);
  add_cell_face_infos(cell_face_info_by_section,  in_r_faces_info,  cell_section_intervals, cell_section_types);
}


template<class I> auto
compute_cell_face(tree_range& cell_sections, const in_ext_faces_by_section<I>& unique_faces_sections, I first_interior_face_id, MPI_Comm comm) {
  int n_cell_section = cell_sections.size();
  std::vector<cell_face_info<I>> cell_face_info_by_section(n_cell_section);

  I current_last_face_id = first_interior_face_id;
  for(const auto& fs : unique_faces_sections) {
    fill_cell_face_info(cell_face_info_by_section,cell_sections,fs,current_last_face_id, comm);

    current_last_face_id += std_e::all_reduce(fs.in.size(),MPI_SUM,comm);
  }

  return cell_face_info_by_section;
}


template<class I> auto
append_cell_face_info(tree& vol_section, std::vector<I>&& cell_indices, std::vector<I>&& cell_face_ids, MPI_Comm comm) {
  auto partial_dist = ElementDistribution<I>(vol_section);

  auto dist_cell_face = maia::distribution_from_partial(partial_dist,comm);
  std_e::scale(dist_cell_face,number_of_faces(element_type(vol_section)));

  std_e::dist_array<I> cell_face(dist_cell_face,comm);
  auto sp = create_exchange_protocol(dist_cell_face,std::move(cell_indices));
  std_e::scatter(sp,cell_face_ids,cell_face);

  std::vector<I> connec_array(begin(cell_face.local()),end(cell_face.local()));
  emplace_child(vol_section,cgns::new_DataArray("CellFace",std::move(connec_array)));
}


template<class I> auto
add_cell_face_connectivities(tree_range& cell_sections, const in_ext_faces_by_section<I>& unique_faces_sections, I first_interior_face_id, MPI_Comm comm) -> void {
  int n_cell_section = cell_sections.size();

  // for each cell section, find its face ids
  auto cell_face_info_by_section = compute_cell_face(cell_sections,unique_faces_sections,first_interior_face_id,comm);

  // add CellFace node to cell sections
  for (int i=0; i<n_cell_section; ++i) {
    auto& [cell_indices,cell_face_ids] = cell_face_info_by_section[i];
    append_cell_face_info(cell_sections[i],std::move(cell_indices),std::move(cell_face_ids),comm);
  }
}


// Explicit instanciations of functions defined in this .cpp file
template auto add_cell_face_connectivities(cgns::tree_range& cell_sections, const in_ext_faces_by_section<I4>& unique_faces_sections, I4 first_interior_face_id, MPI_Comm comm) -> void;
template auto add_cell_face_connectivities(cgns::tree_range& cell_sections, const in_ext_faces_by_section<I8>& unique_faces_sections, I8 first_interior_face_id, MPI_Comm comm) -> void;


} // maia
#endif // C++>17
