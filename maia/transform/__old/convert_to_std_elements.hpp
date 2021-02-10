#pragma once

#include "cpp_cgns/cgns.hpp"

namespace cgns {

auto
sorted_nfaces_to_std_elements(tree& b) -> void;

auto
sort_nfaces_by_element_type(tree& b) -> void;
auto
convert_zone_to_std_elements(tree& z) -> void;

} // cgns
