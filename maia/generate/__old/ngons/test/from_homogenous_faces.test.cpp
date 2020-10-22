#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/__old/ngons/from_homogenous_faces.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"
#include "range/v3/range/conversion.hpp"

TEST_CASE("convert to ngons") {
  std::vector<cgns::quad_4<int32_t>> cs = {
    {42,43,44,45},
    { 5, 4, 3, 2},
    { 0, 2, 4, 6}
  };

  SUBCASE("interleaved") {
    std::vector<int32_t> ngons = convert_to_interleaved_ngons(cs) | ranges::to<std::vector>;

    std::vector<int32_t> expected_ngons = {
      4, 42,43,44,45,
      4,  5, 4, 3, 2,
      4,  0, 2, 4, 6
    };
    
    CHECK( ngons == expected_ngons );
  }

  SUBCASE("non-interleaved") {
    auto res = convert_to_ngons(cs); // NOTE: as of GCC 9.2, structured binding are not working as expected (bug?)
    std::vector<int32_t> start_indices = res.first  | ranges::to<std::vector>;
    std::vector<int32_t> ngon_cs       = res.second | ranges::to<std::vector>;

    std::vector<int32_t> expected_start_indices = { 0,4,8 };
    std::vector<int32_t> expected_ngon_cs = {
      42,43,44,45,
       5, 4, 3, 2,
       0, 2, 4, 6
    };
    
    CHECK( start_indices == expected_start_indices );
    CHECK( ngon_cs == expected_ngon_cs );
  }
}
