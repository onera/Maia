#pragma once


#include "cpp_cgns/cgns.hpp"
#include "mpi.h"


namespace cgns {

auto merge_by_elt_type(tree& b, MPI_Comm comm) -> void;

} // cgns
