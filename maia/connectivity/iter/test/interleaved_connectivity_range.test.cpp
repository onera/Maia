#include "std_e/unit_test/doctest.hpp"

#include "maia/connectivity/iter/interleaved_connectivity_range.hpp"
#include "maia/connectivity/iter/interleaved_connectivity_random_access_range.hpp"
#include "maia/connectivity/iter/poly_elt_t_kind.hpp"

using namespace maia;

TEST_CASE("interleaved ngon connectivity") {
  const std::vector<int> cs = {3, 100,101,102,    4, 142,143,144,145,    2, 44,45};

  SUBCASE("vertices") {
    auto vtx_range = make_interleaved_connectivity_vertex_range<interleaved_polygon_kind>(cs);

    std::vector<int> vertices;
    for (auto vertex: vtx_range) {
      vertices.push_back(vertex);
    }

    const std::vector<int> expected_vertices = {100,101,102,    142,143,144,145,    44,45};
    CHECK( vertices == expected_vertices );
  }


  SUBCASE("index_table") {
    auto cs_fwd_range = make_interleaved_connectivity_range<interleaved_polygon_kind>(cs);

    auto idx_table = index_table(cs_fwd_range);

    CHECK( idx_table.size() == 4 );
    CHECK( idx_table[0] == 0 );
    CHECK( idx_table[1] == 4 );
    CHECK( idx_table[2] == 9 );
    CHECK( idx_table[3] == 12);
  }


  SUBCASE("random_iterator") {
    using random_it = interleaved_connectivity_random_access_iterator<const int,interleaved_polygon_kind>;

    auto cs_fwd_range = make_interleaved_connectivity_range<interleaved_polygon_kind>(cs);
    auto idx_table = index_table(cs_fwd_range);

    int first_pos = 0;
    auto cs_random_it = random_it(cs.data(),idx_table,first_pos);

    auto c_0 = *cs_random_it;
    CHECK( c_0.size() == 3 );
    CHECK( c_0[0] == 100 );
    CHECK( c_0[1] == 101 );
    CHECK( c_0[2] == 102 );

    cs_random_it += 2;
    auto c_2 = *cs_random_it;
    CHECK( c_2.size() == 2 );
    CHECK( c_2[0] == 44 );
    CHECK( c_2[1] == 45 );

    cs_random_it += 0;
    auto still_c_2 = *cs_random_it;
    CHECK( still_c_2.size() == 2 );
    CHECK( still_c_2[0] == 44 );
    CHECK( still_c_2[1] == 45 );

    auto c_1 = *(--cs_random_it);
    CHECK( c_1.size() == 4 );
    CHECK( c_1[0] == 142 );
    CHECK( c_1[1] == 143 );
    CHECK( c_1[2] == 144 );
    CHECK( c_1[3] == 145 );
  }

  SUBCASE("random_access_range") {
    auto random_access_range = make_interleaved_connectivity_random_access_range<interleaved_polygon_kind>(cs);

    REQUIRE( random_access_range.size() == 3 );

    auto c_0 = random_access_range[0];
    CHECK( c_0.size() == 3 );
    CHECK( c_0[0] == 100 );
    CHECK( c_0[1] == 101 );
    CHECK( c_0[2] == 102 );

    auto c_1 = random_access_range[1];
    CHECK( c_1.size() == 4 );
    CHECK( c_1[0] == 142 );
    CHECK( c_1[1] == 143 );
    CHECK( c_1[2] == 144 );
    CHECK( c_1[3] == 145 );

    auto c_2 = random_access_range[2];
    CHECK( c_2.size() == 2 );
    CHECK( c_2[0] == 44 );
    CHECK( c_2[1] == 45 );
  }
};
