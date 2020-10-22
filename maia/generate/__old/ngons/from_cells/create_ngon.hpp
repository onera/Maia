#pragma once


#include <cstdint>
#include "cpp_cgns/sids/creation.hpp"


template<class I> class faces_container; // TODO in namespace cgns
namespace cgns {


auto
create_ngon(const faces_container<std::int32_t>& all_faces, std::int32_t first_ngon_id, factory& F) -> tree;


} // cgns
