#include "std_e/unit_test/doctest.hpp"

#include "maia/connectivity/iter/heterogenous_connectivity_ref.hpp"
#include "maia/connectivity/iter/poly_elt_t_kind.hpp"

using namespace std;
using namespace maia;

using con_ref_type = heterogenous_connectivity_ref<int,int,interleaved_polygon_kind>;
using con_const_ref_type = heterogenous_connectivity_ref<const int,const int,interleaved_polygon_kind>;

TEST_CASE("connectivity_ref") {
  int connectivity_elt_t = 3;
  vector<int> vertices = {1,2,3};
  int* v_ptr = vertices.data();
  const int* v_const_ptr = vertices.data();

  con_const_ref_type con_const_ref(connectivity_elt_t,v_const_ptr);
  con_ref_type       con_ref_type (connectivity_elt_t,v_ptr      );

  SUBCASE("basic_tests") {
    CHECK( con_ref_type.elt_t() == 3 );
    CHECK( con_ref_type.size()  == 3 );
  }

  SUBCASE("equality") {
    int same_connectivity_elt_t = 3;
    vector<int> same_vertices = {1,2,3};
    con_const_ref_type same(same_connectivity_elt_t,same_vertices.data());
    CHECK( con_ref_type == same );
    CHECK_FALSE( con_ref_type < same );

    int different_connectivity_elt_t = 2;
    con_const_ref_type different_type(different_connectivity_elt_t,same_vertices.data());
    CHECK( con_ref_type != different_type );
    CHECK( different_type < con_ref_type );

    vector<int> different_vertices = {2,3,1};
    con_const_ref_type different(same_connectivity_elt_t,different_vertices.data());
    CHECK( con_ref_type != different );
    CHECK( con_ref_type < different );

    vector<int> inferior_vertices = {1,2,0};
    con_const_ref_type inferior(same_connectivity_elt_t,inferior_vertices.data());
    CHECK( con_ref_type != inferior );
    CHECK( inferior < con_ref_type );
  }


  SUBCASE("read") {
    CHECK( con_const_ref[0] == 1 );
    CHECK( con_const_ref[1] == 2 );
    CHECK( con_const_ref[2] == 3 );
  }
      
  SUBCASE("write") {
    con_ref_type[0] = 10;
    con_ref_type[1] = 11;
    con_ref_type[2] = 12;

    CHECK( con_const_ref[0] == 10 );
    CHECK( con_const_ref[1] == 11 );
    CHECK( con_const_ref[2] == 12 );
  }

  SUBCASE("begin_and_end") {
    CHECK( con_ref_type.begin() == vertices.data()  );
    CHECK( con_ref_type.end()   == vertices.data()+3);
  }
}
