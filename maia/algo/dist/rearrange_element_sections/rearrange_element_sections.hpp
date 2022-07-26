#pragma once


#include "cpp_cgns/cgns.hpp"
#include "mpi.h"


namespace maia {

auto rearrange_element_sections(cgns::tree& b, MPI_Comm comm) -> void;

} // maia
