#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/partition_with_boundary_first/boundary_ngons_at_beginning.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/sids/creation.hpp"

using namespace cgns;
using std::vector;


TEST_CASE("boundary_ngons_at_beginning") {
  cgns_allocator alloc; // allocates and owns memory
  factory F(&alloc);

  // ngon connectivities
  I4 first_ngon_elt = 6;
  I4 last_ngon_elt = 9;
  auto ngon_cs = make_cgns_vector<I4>(
    { 3,   1, 2, 3,
      4,  10,11,12,13, 
      3,   5, 4, 3,
      3,   1, 8, 9 },
    alloc
  );
  auto parent_elts = make_md_array<I4>(
    { {1, 4},
      {0, 8},
      {0, 0},
      {3, 1} },
    alloc
  );

  tree ngons = F.newNgonElements(
    "Ngons",
    std_e::make_span(ngon_cs),
    first_ngon_elt,last_ngon_elt
  );
  emplace_child(ngons,F.new_DataArray("ParentElements", view_as_node_value(parent_elts)));


  SUBCASE("boundary/interior_permutation") {
    auto [ngon_permutation,partition_index] = boundary_interior_permutation(parent_elts);

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
      apply_ngon_permutation_to_parent_elts(parent_elts,my_permutation);

      auto expected_parent_elts = make_md_array<I4>(
        { {0, 8},
          {0, 0},
          {1, 4},
          {3, 1} },
        alloc
      );
      CHECK( parent_elts == expected_parent_elts );
    }
  }

  SUBCASE("permute_boundary_ngons_at_beginning") {
    auto ngon_permutation = permute_boundary_ngons_at_beginning(ngons,F);

    vector<I4> expected_ngon_permutation = {1,2,0,3};
    CHECK( ngon_permutation == expected_ngon_permutation );


    auto expected_ngon_cs = make_cgns_vector<I4>(
      { 4,  10,11,12,13, 
        3,   5, 4, 3,
        3,   1, 2, 3,
        3,   1, 8, 9 },
      alloc
    );
    CHECK( ngon_cs == expected_ngon_cs );


    auto expected_parent_elts = make_md_array<I4>(
        { {0, 8},
          {0, 0},
          {1, 4},
          {3, 1} },
      alloc
    );
    CHECK( parent_elts == expected_parent_elts );

    CHECK( ElementSizeBoundary<I4>(ngons) == 2 );

    tree_range partition_index_nodes = get_children_by_label(ngons,"UserDefinedData_t");
    REQUIRE( partition_index_nodes.size() == 1 );
    CHECK( name(partition_index_nodes[0]) == ".#PartitionIndex" );

    I4* ord_ptr = (I4*)value(partition_index_nodes[0]).data;
    CHECK( *ord_ptr == 1+4 + 1+3 );
  }
}
