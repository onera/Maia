#include "std_e/unit_test/doctest.hpp"

#include "maia/connectivity/iter/connectivity_range.hpp"
#include "maia/connectivity/iter/test/test_utils.hpp"

using namespace std;

using con_view_type = connectivity_ref<int,my_connectivity_kind>;
using con_const_view_type = connectivity_ref<const int,my_connectivity_kind>;

using con_it_type = connectivity_iterator<int,my_connectivity_kind>;
using con_const_it_type = connectivity_iterator<const int,my_connectivity_kind>;
using con_range = connectivity_range<vector<int>,my_connectivity_kind>;

TEST_CASE("connectivity_range") {
  std::vector<int> c = {3,4,5,  6,10,12};

  SUBCASE("test__homogeneous_connectivities,const_iterator") {
    con_const_it_type const_it(c.data());

    con_const_view_type expected_con0(c.data());
    con_const_view_type expected_con1(c.data()+3);

    CHECK( *const_it == expected_con0 );

    ++const_it;

    CHECK( *const_it == expected_con1 );
  }

  SUBCASE("test__homogeneous_connectivities,iterator") {
    con_it_type it = {c.data()};

    auto con0 = *it;

    con0[0] = 42;
    con0[1] = 43;
    con0[2] = 44;

    CHECK( c[0] == 42 );
    CHECK( c[1] == 43 );
    CHECK( c[2] == 44 );


    ++it;


    auto con1 = *it;

    con1[0] = 100;
    con1[1] = 101;
    con1[2] = 102;

    CHECK( c[3] == 100 );
    CHECK( c[4] == 101 );
    CHECK( c[5] == 102 );
  }

  SUBCASE("test__homogeneous_connectivities,access") {
    auto c_range = make_connectivity_range<my_connectivity_kind>(c);

    con_const_view_type expected_con0(c.data());
    con_const_view_type expected_con1(c.data()+3);

    CHECK( c_range[0] == expected_con0 );
    CHECK( c_range[1] == expected_con1 );
  }
};
