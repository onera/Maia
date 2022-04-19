#pragma once


#include "cpp_cgns/cgns.hpp"
#include <mpi.h>


namespace maia {


auto generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void;


} // maia
