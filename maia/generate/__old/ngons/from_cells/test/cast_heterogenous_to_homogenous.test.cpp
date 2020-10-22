#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/ngons/from_cells/cast_heterogenous_to_homogenous.hpp"

TEST_CASE("cast_heterogenous_to_homogenous") {
  int quad_4_elt_t = cgns::QUAD_4;
  std::vector<int> quad_data = {3,9,12,1};

  heterogenous_connectivity_ref<int,int,cgns::mixed_kind> quad_viewed_as_mixed(quad_4_elt_t,quad_data.data());

  auto quad = cgns::cast_as<cgns::QUAD_4>(quad_viewed_as_mixed);

  static_assert(quad.elt_t==cgns::QUAD_4);
  static_assert(quad.size()==4);

  CHECK( quad[0] ==  3 );
  CHECK( quad[1] ==  9 );
  CHECK( quad[2] == 12 );
  CHECK( quad[3] ==  1 );
}
