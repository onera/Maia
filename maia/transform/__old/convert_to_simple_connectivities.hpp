#pragma once

#include "cpp_cgns/cgns.hpp"

namespace cgns {

auto
convert_to_simple_connectivities(tree& b) -> void;

auto
sort_nface_into_simple_connectivities(tree& b) -> void;
auto
convert_zone_to_simple_connectivities(tree& z) -> void;

} // cgns
