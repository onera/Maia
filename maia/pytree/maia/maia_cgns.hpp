#pragma once


#include "cpp_cgns/cgns_fwd.hpp"


namespace maia {


auto is_maia_compliant_zone(const cgns::tree& z) -> bool;


} // maia
