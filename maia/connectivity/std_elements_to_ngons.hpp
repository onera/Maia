#pragma once


#include "cpp_cgns/cgns.hpp"
#include <mpi.h>


namespace maia {


auto replace_faces_by_ngons(cgns::tree& z, MPI_Comm comm) -> void;
auto std_elements_to_ngons(cgns::tree& z, MPI_Comm comm) -> void;


} // maia
