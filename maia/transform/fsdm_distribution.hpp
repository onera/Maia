#pragma once


#include "cpp_cgns/cgns.hpp"
#include "mpi.h"


namespace cgns {

auto add_fsdm_distribution(tree& b, MPI_Comm comm) -> void;

} // cgns
