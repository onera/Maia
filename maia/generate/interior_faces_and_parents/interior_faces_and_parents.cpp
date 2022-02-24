#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"

#include "maia/sids/element_sections.hpp"
#include "maia/sids/maia_cgns.hpp"
#include "maia/utils/log/log.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"
#include "maia/generate/interior_faces_and_parents/merge_unique_faces.hpp"
#include "maia/generate/interior_faces_and_parents/scatter_parents_to_boundary_sections.hpp"
#include "maia/generate/interior_faces_and_parents/add_cell_face_connectivities.hpp"
#include "maia/generate/interior_faces_and_parents/append_interior_faces_sections.hpp"


using namespace cgns;


namespace maia {


template<class I> auto
_generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void {
  STD_E_ASSERT(is_maia_cgns_zone(z));

  // 0. Base queries
  auto elt_sections = element_sections(z);
  auto bnd_face_sections = surface_element_sections(z);
  auto cell_sections = volume_element_sections(z);

  // 1. Generate faces
  auto faces_and_parents_sections = generate_element_faces_and_parents<I>(elt_sections);

  // 2. Make them unique
  I first_3d_elt_id = elements_interval(cell_sections).first();
  auto unique_faces_sections = merge_unique_faces(faces_and_parents_sections,first_3d_elt_id,comm);

  // 3. Send parent info back to original exterior faces
  scatter_parents_to_boundary_sections(bnd_face_sections,unique_faces_sections,comm);

  // 4. Start of interior faces
  //   interior faces will be given an ElementRange just after exterior faces,
  //   thus starting and overlapping with cell ids
  I first_interior_face_id = elements_interval(cell_sections).first();

  // 5. Compute cell_face
  add_cell_face_connectivities(cell_sections,unique_faces_sections,first_interior_face_id,comm);

  // 6. Create new interior faces sections
  append_interior_faces_sections(z,std::move(unique_faces_sections),first_interior_face_id,comm);
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
