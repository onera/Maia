#pragma once


#include "cpp_cgns/cgns.hpp"


namespace maia {

auto indexed_to_interleaved_connectivity(cgns::tree& elt) -> void;
auto interleaved_to_indexed_connectivity(cgns::tree& elt) -> void;

} // maia
