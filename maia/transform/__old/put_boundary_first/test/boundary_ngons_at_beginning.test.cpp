#if __cplusplus > 201703L
#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/put_boundary_first/boundary_ngons_at_beginning.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;
using std::vector;


TEST_CASE("boundary_ngons_at_beginning") {
  // ngons
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
  tree ngons = new_NgonElements(
    "Ngons",
    std::move(ngon_cs),
    first_ngon_elt,last_ngon_elt
  );
  emplace_child(ngons,new_DataArray("ElementStartOffset", std::move(ngon_eso)));
  emplace_child(ngons,new_DataArray("ParentElements", std::move(parent_elts)));
  // nfaces
  I4 first_nface_elt = 10;
  I4 last_nface_elt = 11;
  vector<I4> nface_cs =
    { 6,7,8,9,
      9,8,7 };
  vector<I4> nface_eso = {0,4,7};

  tree nfaces = new_NfaceElements(
    "Nfaces",
    std::move(nface_cs),
    first_nface_elt,last_nface_elt
  );
  emplace_child(nfaces,new_DataArray("ElementStartOffset", std::move(nface_eso)));
  // point lists
  std::vector<I4> pl = {8,9};
  std::vector<std_e::span<I4>> pls = {std_e::make_span(pl)};

  SUBCASE("boundary/interior_permutation") {
    auto pe = ParentElements<I4>(ngons);
    auto [ngon_permutation,partition_index] = maia::boundary_interior_permutation(pe);

    CHECK( partition_index == 2 );

    vector<I4> expected_ngon_permutation = {1,2,0,3};
    CHECK( ngon_permutation == expected_ngon_permutation );
  }

  SUBCASE("permute_boundary_ngons_at_beginning") {
    auto last_exterior = maia::permute_boundary_ngons_at_beginning<I4>(ngons,nfaces,pls);

    CHECK( last_exterior == 2 );
    vector<I4> expected_ngon_cs =
      { 10,11,12,13,
         5, 4, 3,
         1, 2, 3,
         1, 8, 9 };
    vector<I4> expected_ngon_eso = {0,4,7,10,13};
    md_array<I4,2> expected_parent_elts =
        { {0, 8},
          {0, 0},
          {1, 4},
          {3, 1} };
    vector<I4> expected_nface_cs =
      { 8,6,7,9,
        9,7,6 };

    CHECK( ElementConnectivity<I4>(ngons) == expected_ngon_cs );
    CHECK( ElementStartOffset<I4>(ngons) == expected_ngon_eso );
    CHECK( ParentElements<I4>(ngons) == expected_parent_elts );

    CHECK( ElementSizeBoundary(ngons) == 2 );

    CHECK( ElementConnectivity<I4>(nfaces) == expected_nface_cs );

    CHECK( pl == std::vector{7,9} );
  }

  SUBCASE("partition_bnd_faces_by_number_of_vertices") {
    // Not testing this, just using the result
    maia::permute_boundary_ngons_at_beginning<I4>(ngons,nfaces,pls);

    // This is the one we are testing
    auto last_tri_bnd = maia::partition_bnd_faces_by_number_of_vertices<I4>(ngons,nfaces,pls);

    CHECK( last_tri_bnd == 1 );
    vector<I4> expected_ngon_cs =
      {  5, 4, 3,
        10,11,12,13,
         1, 2, 3,
         1, 8, 9 };
    vector<I4> expected_ngon_eso = {0,3,7,10,13};
    md_array<I4,2> expected_parent_elts =
        { {0, 0},
          {0, 8},
          {1, 4},
          {3, 1} };
    vector<I4> expected_nface_cs =
      { 8,7,6,9,
        9,6,7 };

    CHECK( ElementConnectivity<I4>(ngons) == expected_ngon_cs );
    CHECK( ElementStartOffset<I4>(ngons) == expected_ngon_eso );
    CHECK( ParentElements<I4>(ngons) == expected_parent_elts );

    CHECK( ElementConnectivity<I4>(nfaces) == expected_nface_cs );

    CHECK( pl == std::vector{6,9} );
  }
}
#endif // C++>17
