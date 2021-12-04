#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/partition_with_boundary_first/boundary_ngons_at_beginning.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;
using std::vector;


TEST_CASE("boundary_ngons_at_beginning") {
  // ngon connectivities
  I4 first_ngon_elt = 6;
  I4 last_ngon_elt = 9;
  std::vector<I4> ngon_cs =
    { 3,   1, 2, 3,
      4,  10,11,12,13,
      3,   5, 4, 3,
      3,   1, 8, 9 };
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
    I4 my_partition_index = 2;

    SUBCASE("ngons") {
      auto ngon_partition_index = apply_partition_to_ngons(std_e::make_span(ngon_cs),my_permutation,my_partition_index);

      vector<I4> expected_ngon_cs = {
        4,  10,11,12,13,
        3,   5, 4, 3,
        3,   1, 2, 3,
        3,   1, 8, 9
      };
      CHECK( std_e::make_span(ngon_cs) == std_e::make_span(expected_ngon_cs) );

      CHECK( ngon_partition_index == 1+4 + 1+3);
      //                              ^     ^
      //                              |     |
      //                        sz ngon 0 + sz ngon 1
    }
    SUBCASE("parent elts") {
      auto pe_view = make_view(parent_elts);
      apply_ngon_permutation_to_parent_elts(pe_view,my_permutation);

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
    tree& pe_node = emplace_child(ngons,new_DataArray("ParentElements", std::move(parent_elts)));

    auto ngon_permutation = permute_boundary_ngons_at_beginning(ngons);

    vector<I4> expected_ngon_permutation = {1,2,0,3};
    CHECK( ngon_permutation == expected_ngon_permutation );


    std::vector<I4> expected_ngon_cs =
      { 4,  10,11,12,13,
        3,   5, 4, 3,
        3,   1, 2, 3,
        3,   1, 8, 9 };
    CHECK( ElementConnectivity<I4>(ngons) == expected_ngon_cs );


    md_array<I4,2> expected_parent_elts =
        { {0, 8},
          {0, 0},
          {1, 4},
          {3, 1} };
    CHECK( get_value<I4,2>(pe_node) == expected_parent_elts );

    CHECK( ElementSizeBoundary(ngons) == 2 );

    tree_range partition_index_nodes = get_children_by_label(ngons,"UserDefinedData_t");
    REQUIRE( partition_index_nodes.size() == 1 );
    CHECK( name(partition_index_nodes[0]) == ".#PartitionIndex" );
    CHECK( value(partition_index_nodes[0]).rank() == 1 );
    CHECK( value(partition_index_nodes[0]).extent(0) == 1 );
    CHECK( value(partition_index_nodes[0])(0) == 1+4 + 1+3 );
  }
}
