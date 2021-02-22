#pragma once


#include "cpp_cgns/cgns.hpp"


namespace cgns {

auto split_boundary_subzones_according_to_bcs(tree& b) -> void;

} // cgns
