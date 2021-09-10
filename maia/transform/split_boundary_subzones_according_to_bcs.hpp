#pragma once


#include "cpp_cgns/cgns.hpp"
#include <mpi.h>


namespace maia {

auto split_boundary_subzones_according_to_bcs(cgns::tree& b, MPI_Comm comm) -> void;

} // maia
