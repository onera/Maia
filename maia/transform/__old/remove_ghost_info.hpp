#pragma once

#include "cpp_cgns/cgns.hpp"
#include "mpi.h"
#include "maia/transform/__old/donated_point_lists.hpp"

namespace cgns {

auto remove_ghost_info(tree& b, MPI_Comm comm) -> void;

auto remove_ghost_info_from_zone(tree& z, donated_point_lists& plds) -> void;

} // cgns
