#pragma once


#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/struct/faces_and_parents_by_section.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/struct/in_ext_faces_by_section.hpp"
#include "mpi.h"


namespace maia {


template<class I> auto
merge_unique_faces(faces_and_parents_by_section<I>& faces_and_parents_sections, I first_3d_elt_id, MPI_Comm comm)
 -> in_ext_faces_by_section<I>;


} // maia
