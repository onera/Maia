#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

using namespace std;
using namespace cgns;


TEST_CASE("partition_with_boundray_vertices") {
// tri3 connectivities
  std::vector<I4> tri_cs =
    {  1, 2, 3,
      42,43,44,
      44,43, 3  };

// tet4 connectivities
  std::vector<I4> tet_cs =
    {  1, 2, 3, 4,
      42,43,44,45  };

// ngon connectivities
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

  SUBCASE("ngon_boundary_vertices") {
    auto boundary_vertices = ngon_boundary_vertices(std_e::make_span(ngon_cs),std_e::make_view(parent_elts));

    vector<I4> expected_boundary_vertices = {
      10,11,12,13,
      5, 4, 3
    };

    CHECK( boundary_vertices == expected_boundary_vertices );
  }

  SUBCASE("get_ordered_boundary_vertex_ids") {
    // construction of elements
    tree tris = new_Elements(
      "Tri",
      cgns::TRI_3,
      std::move(tri_cs),
      1,3
    );
    tree tets = new_Elements(
      "Tet",
      cgns::TETRA_4,
      std::move(tet_cs),
      4,5
    );
    tree ngons = new_NgonElements(
      "Ngons",
      std::move(ngon_cs),
      6,9
    );
    emplace_child(ngons,new_DataArray("ParentElements", std::move(parent_elts)));

    tree_range zone_elements = {tris,tets,ngons};
    auto boundary_vertices = get_ordered_boundary_vertex_ids(zone_elements);

    vector<I4> expected_boundary_vertices = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 42, 43, 44 };

    CHECK( boundary_vertices == expected_boundary_vertices );
  }
};
