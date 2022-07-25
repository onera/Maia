#pragma once


#include "cpp_cgns/cgns.hpp"
#include "maia/algo/dist/elements_to_ngons/interior_faces_and_parents/struct/in_ext_faces_by_section.hpp"
#include "mpi.h"


namespace maia {


template<class I> auto
add_cell_face_connectivities(cgns::tree_range& cell_sections, const in_ext_faces_by_section<I>& unique_faces_sections, I first_interior_face_id, MPI_Comm comm) -> void;


} // maia
