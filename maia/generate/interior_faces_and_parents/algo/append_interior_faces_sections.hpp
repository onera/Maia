#pragma once


#include "cpp_cgns/cgns.hpp"
#include "maia/generate/interior_faces_and_parents/struct/in_ext_faces_by_section.hpp"
#include "mpi.h"


using namespace cgns;


namespace maia {


template<class I> auto
append_interior_faces_sections(cgns::tree& z, in_ext_faces_by_section<I>&& faces_sections, I first_interior_face_id, MPI_Comm comm) -> I;


} // maia
