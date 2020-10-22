#include "std_e/unit_test/doctest.hpp"

#include "maia/connectivity/iter_cgns/range.hpp"


using MyTestConnectivity = std::vector<int>;


TEST_CASE("ngon fwd iterator") {
  const std::vector<int> ngons = {
    2, 42,43,
    4, 110,101,99,98,
    3, 5,25,50
  };

  std::vector<MyTestConnectivity> cs;
  for (const auto& ngon : cgns::interleaved_ngon_range(ngons)) {
    MyTestConnectivity c;
    for (auto vertex : ngon) {
      c.push_back(vertex);
    }
    cs.push_back(c);
  }

  REQUIRE( cs.size() == 3 );

  REQUIRE( cs[0].size() == 2 );
  CHECK( cs[0][0] == 42 );
  CHECK( cs[0][1] == 43 );

  REQUIRE( cs[1].size() == 4 );
  CHECK( cs[1][0] == 110 );
  CHECK( cs[1][1] == 101 );
  CHECK( cs[1][2] ==  99 );
  CHECK( cs[1][3] ==  98 );

  REQUIRE( cs[2].size() == 3 );
  CHECK( cs[2][0] ==  5 );
  CHECK( cs[2][1] == 25 );
  CHECK( cs[2][2] == 50 );
}
