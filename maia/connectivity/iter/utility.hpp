#pragma once

#include <vector>
#include "maia/connectivity/iter/connectivity.hpp"
#include "std_e/algorithm/algorithm.hpp"


namespace maia {


template<class I, class CK> inline auto
offset_vertices_ids(std::vector<connectivity<I,CK>>& cs, I value) {
  for (auto& c : cs) {
    std_e::offset(c,value);
  }
}


} // maia
