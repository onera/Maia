#pragma once


#include "cpp_cgns/cgns.hpp"


namespace maia {

auto ngon_new_to_old(cgns::tree& b) -> void;
auto ngon_old_to_new(cgns::tree& b) -> void;

} // maia
