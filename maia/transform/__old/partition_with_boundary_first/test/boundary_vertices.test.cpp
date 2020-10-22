#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

using namespace std;
using namespace cgns;


TEST_CASE("partition_with_boundray_vertices") {
  cgns_allocator alloc; // allocates and owns memory
  factory F(&alloc);

// tri3 connectivities
  auto tri_cs = make_cgns_vector<I4>(
    {  1, 2, 3,
      42,43,44,
      44,43, 3  },
    alloc
  );

// tet4 connectivities
  auto tet_cs = make_cgns_vector<I4>(
    {  1, 2, 3, 4,
      42,43,44,45  },
    alloc
  );

// ngon connectivities
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

// construction of elements
  tree tris = F.newElements(
    "Tri",
    cgns::TRI_3,
    std_e::make_span(tri_cs),
    1,3
  );
  tree tets = F.newElements(
    "Tet",
    cgns::TETRA_4,
    std_e::make_span(tet_cs),
    4,5
  );
  tree ngons = F.newNgonElements(
    "Ngons",
    std_e::make_span(ngon_cs),
    6,9
  );
  emplace_child(ngons,F.newDataArray("ParentElements", view_as_node_value(parent_elts)));


  SUBCASE("ngon_boundary_vertices") {
    auto boundary_vertices = ngon_boundary_vertices(std_e::make_span(ngon_cs),std_e::make_view(parent_elts));

    vector<I4> expected_boundary_vertices = {
      10,11,12,13, 
      5, 4, 3
    };

    CHECK( boundary_vertices == expected_boundary_vertices );
  }

  SUBCASE("get_ordered_boundary_vertex_ids") {
    tree_range zone_elements = {tris,tets,ngons};
    auto boundary_vertices = get_ordered_boundary_vertex_ids(zone_elements);

    vector<I4> expected_boundary_vertices = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 42, 43, 44 };

    CHECK( boundary_vertices == expected_boundary_vertices );
  }
};
