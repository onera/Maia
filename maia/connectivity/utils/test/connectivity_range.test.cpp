#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind.hpp"

#include "cpp_cgns/sids/creation.hpp"
#include "maia/connectivity/utils/connectivity_range.hpp"

using cgns::I4;
using cgns::md_array;
using cgns::tree;
using std::vector;

TEST_CASE("connectivity_range") {
  I4 first_ngon_elt = 6;
  I4 last_ngon_elt = 9;
  vector<I4> ngon_cs =
    {  1, 2, 3,
      10,11,12,13,
       5, 4, 3,
       1, 8, 9 };
  vector<I4> ngon_eso = {0,3,7,10,13};
  md_array<I4,2> parent_elts =
    { {1, 4},
      {0, 8},
      {0, 0},
      {3, 1} };

  tree ngons = cgns::new_NgonElements(
    "Ngons",
    std::move(ngon_cs),
    first_ngon_elt,last_ngon_elt
  );
  emplace_child(ngons,cgns::new_DataArray("ElementStartOffset", std::move(ngon_eso)));
  emplace_child(ngons,cgns::new_DataArray("ParentElements", std::move(parent_elts)));

  SUBCASE("make_connectivity_range") {
    auto face_vtx = maia::make_connectivity_range<I4>(ngons);
    CHECK( face_vtx.size() == 4 );
    CHECK( face_vtx[0] == vector{1,2,3} );
    CHECK( face_vtx[1] == vector{10,11,12,13} );
    CHECK( face_vtx[2] == vector{5,4,3} );
    CHECK( face_vtx[3] == vector{1,8,9} );
  }

  SUBCASE("make_connectivity_subrange") {
    auto face_vtx = maia::make_connectivity_subrange<I4>(ngons,1,3);
    CHECK( face_vtx.size() == 2 );
    CHECK( face_vtx[0] == vector{10,11,12,13} );
    CHECK( face_vtx[1] == vector{5,4,3} );
  }
}
#endif // C++>17
