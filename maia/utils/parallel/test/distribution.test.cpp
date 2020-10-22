#include "std_e/unit_test/doctest.hpp"
#include "maia/utils/parallel/distribution.hpp"

TEST_CASE("compute_multi_range_from_count") {
  std::vector v = {3,4,5,2};
  compute_multi_range_from_count(v,10);

  CHECK( v == std::vector{10,13,17,22} );
}
