#pragma once


#include "cpp_cgns/cgns.hpp"
#include <mpi.h>


namespace maia {


auto std_elements_to_ngons(cgns::tree& z, MPI_Comm comm) -> void;


} // maia
