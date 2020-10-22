#pragma once

#include <vector>
#include "std_e/future/span.hpp"
#include "cpp_cgns/sids/Building_Block_Structure_Definitions.hpp"

namespace cgns {

// PointList values coming from a receiver zone (that this receiver zone stores in a GridConnectivity)
// TODO multi_range
struct donated_point_list {
  std::string receiver_z_name;
  cgns::GridLocation_t loc;
  std_e::span<I4> pl;
};
auto eq_receiver_zone = [](const donated_point_list& x, const donated_point_list& y) {
  return x.receiver_z_name == y.receiver_z_name ;
};
auto less_receiver_zone = [](const donated_point_list& x, const donated_point_list& y) {
  return x.receiver_z_name < y.receiver_z_name ;
};

using donated_point_lists = std::vector<donated_point_list>;

} // cgns
