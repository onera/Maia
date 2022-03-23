#pragma once

#include "cpp_cgns/cgns.hpp"

namespace maia {


auto
convert_zone_to_std_elements(cgns::tree& z) -> void;


} // maia
