#include "std_e/unit_test/doctest.hpp"

#include "maia/connectivity/iter/poly_elt_t_reference.hpp"
#include "maia/connectivity/iter_cgns/connectivity_kind.hpp"
#include "maia/connectivity/iter/heterogenous_connectivity_ref.hpp"

using namespace maia;

TEST_CASE("example use of poly_elt_t_reference and heterogenous_connectivity_ref: indexed_poly connectivity range copy") {
  using CC_ref = heterogenous_connectivity_ref<const int,const int,indexed_polygon_kind>;
  using C_ref = heterogenous_connectivity_ref<int,int,indexed_polygon_kind>;

// input
  // creation
  const std::vector<int> c_polygon_offsets = { 0            , 4       , 7 };
  const std::vector<int> c_polygon_cs      = { 110,101,99,98, 5,25,50 };

  // references to the connecvitivities
  poly_elt_t_reference<const int> c_elt_t_ref_0(&c_polygon_offsets[0]);
  poly_elt_t_reference<const int> c_elt_t_ref_1(&c_polygon_offsets[1]);

  CC_ref c_connec_ref_0 = {c_elt_t_ref_0,&c_polygon_cs[c_polygon_offsets[0]]};
  CC_ref c_connec_ref_1 = {c_elt_t_ref_1,&c_polygon_cs[c_polygon_offsets[1]]};

// output
  // initialize with size
  std::vector<int> polygon_offsets(c_polygon_offsets.size());
  std::vector<int> polygon_cs(c_polygon_cs.size());
  // the first offset is always 0
  polygon_offsets[0] = 0;

  // first connectivity
  /// ref to output first connectivity
  poly_elt_t_reference<int> elt_t_ref_0(&polygon_offsets[0]);
  C_ref connec_ref_0 = {elt_t_ref_0,&polygon_cs[polygon_offsets[0]]};
  /// assign first input to first output
  connec_ref_0 = c_connec_ref_0;
  /// check that the copy is correct
  CHECK( polygon_offsets[0] == 0 );
  CHECK( polygon_offsets[1] == 4 );
  CHECK( polygon_cs[0] == 110 );
  CHECK( polygon_cs[1] == 101 );
  CHECK( polygon_cs[2] ==  99 );
  CHECK( polygon_cs[3] ==  98 );

  // second connectivity
  /// ref to output second connectivity
  poly_elt_t_reference<int> elt_t_ref_1(&polygon_offsets[1]);
  C_ref connec_ref_1 = {elt_t_ref_1,&polygon_cs[polygon_offsets[1]]};
  /// assign first second to second output
  connec_ref_1 = c_connec_ref_1;
  /// check that the copy is correct
  CHECK( polygon_offsets[0] == 0 );
  CHECK( polygon_offsets[1] == 4 );
  CHECK( polygon_offsets[2] == 7 );
  CHECK( polygon_cs[4] ==  5 );
  CHECK( polygon_cs[5] == 25 );
  CHECK( polygon_cs[6] == 50 );

  // check that everything is correct
  CHECK( polygon_offsets == c_polygon_offsets );
  CHECK( polygon_cs == c_polygon_cs );
}
