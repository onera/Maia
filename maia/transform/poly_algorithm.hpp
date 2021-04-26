#pragma once


#include "cpp_cgns/cgns.hpp"


namespace maia {

auto ngon_new_to_old(cgns::tree& b) -> void;
auto sids_conforming_ngon_nface(cgns::tree& b) -> void;

} // maia
