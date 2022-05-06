#pragma once

#include "cpp_cgns/cgns.hpp"
#include "mpi.h"
#include "maia/__old/transform/donated_point_lists.hpp"

namespace maia {

auto remove_ghost_info(cgns::tree& b, MPI_Comm comm) -> void;

auto remove_ghost_info_from_zone(cgns::tree& z, donated_point_lists& plds) -> void;

} // maia
