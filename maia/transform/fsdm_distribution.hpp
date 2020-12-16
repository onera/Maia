#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "mpi.h"


namespace cgns {

auto add_fsdm_distribution(tree& b, factory F, MPI_Comm comm) -> void;

} // cgns
