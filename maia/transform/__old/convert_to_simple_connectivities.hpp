#pragma once

#include "cpp_cgns/cgns.hpp"

namespace cgns {

// Fwd decl
class factory;

auto
convert_to_simple_connectivities(tree& b, factory F) -> void;

auto
sort_nface_into_simple_connectivities(tree& b, factory F) -> void;
auto
convert_zone_to_simple_connectivities(tree& z, factory F) -> void;

} // cgns
