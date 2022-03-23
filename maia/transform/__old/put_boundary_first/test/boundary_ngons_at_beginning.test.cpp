#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/put_boundary_first/boundary_ngons_at_beginning.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;
using std::vector;


TEST_CASE("boundary_ngons_at_beginning") {
  // ngon connectivities
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


  SUBCASE("boundary/interior_permutation") {
    auto pe_view = make_view(parent_elts);
    auto [ngon_permutation,partition_index] = boundary_interior_permutation(pe_view);

    CHECK( partition_index == 2 );

    vector<I4> expected_ngon_permutation = {1,2,0,3};
    CHECK( ngon_permutation == expected_ngon_permutation );
  }

  SUBCASE("apply partition") {
    vector<I4> my_permutation = {1,2, 0,3};

    SUBCASE("ngons") {
      apply_partition_to_ngons(std_e::make_span(ngon_cs),std_e::make_span(ngon_eso),my_permutation);

      vector<I4> expected_ngon_cs = {
        10,11,12,13,
         5, 4, 3,
         1, 2, 3,
         1, 8, 9
      };
      vector<I4> expected_ngon_eso = {0,4,7,10,13};
      CHECK( std_e::make_span(ngon_cs) == std_e::make_span(expected_ngon_cs) ); // TODO remove make span
      CHECK( std_e::make_span(ngon_cs) == std_e::make_span(expected_ngon_cs) );
    }
    SUBCASE("parent elts") {
      auto pe_view = make_view(parent_elts);
      apply_permutation_to_parent_elts(pe_view,my_permutation);

      md_array<I4,2> expected_parent_elts =
        { {0, 8},
          {0, 0},
          {1, 4},
          {3, 1} };
      CHECK( parent_elts == expected_parent_elts );
    }
  }

  SUBCASE("permute_boundary_ngons_at_beginning") {
    tree ngons = new_NgonElements(
      "Ngons",
      std::move(ngon_cs),
      first_ngon_elt,last_ngon_elt
    );
    emplace_child(ngons,new_DataArray("ElementStartOffset", std::move(ngon_eso)));
    emplace_child(ngons,new_DataArray("ParentElements", std::move(parent_elts)));

    auto ngon_permutation = permute_boundary_ngons_at_beginning<I4>(ngons);

    vector<I4> expected_ngon_permutation = {1,2,0,3};
    CHECK( ngon_permutation == expected_ngon_permutation );


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

    CHECK( ElementConnectivity<I4>(ngons) == expected_ngon_cs );
    CHECK( ElementStartOffset<I4>(ngons) == expected_ngon_eso );
    CHECK( ParentElements<I4>(ngons) == expected_parent_elts );

    CHECK( ElementSizeBoundary(ngons) == 2 );
  }
}
