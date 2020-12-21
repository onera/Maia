#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/creation.hpp"


namespace cgns {

auto gcs_only_for_ghosts(tree& b, factory F) -> void;

} // cgns
