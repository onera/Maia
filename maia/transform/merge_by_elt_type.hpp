#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "mpi.h"


namespace cgns {

auto merge_by_elt_type(tree& b, factory F, MPI_Comm comm) -> void;

} // cgns
