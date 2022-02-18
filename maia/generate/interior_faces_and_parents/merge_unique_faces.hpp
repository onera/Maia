#pragma once


#include "maia/generate/interior_faces_and_parents/faces_and_parents_by_section.hpp"
#include "maia/generate/interior_faces_and_parents/in_ext_faces_with_parents.hpp"
#include "mpi.h"


namespace maia {


template<class I> auto
merge_unique_faces(faces_and_parents_by_section<I>& faces_and_parents_sections, I n_vtx, I first_3d_elt_id, MPI_Comm comm)
 -> std::array<in_ext_faces_with_parents<I>,cgns::n_face_types>;


} // maia
