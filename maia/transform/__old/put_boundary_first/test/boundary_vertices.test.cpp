#include "std_e/unit_test/doctest.hpp"

#include "maia/transform/__old/put_boundary_first/boundary_vertices.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "std_e/data_structure/block_range/vblock_range.hpp"

using namespace std;
using namespace cgns;
using namespace maia;


TEST_CASE("partition_with_boundary_vertices") {
// tri3 connectivities
  vector<I4> tri_cs =
    {  1, 2, 3,
      42,43,44,
      44,43, 3  };

// tet4 connectivities
  vector<I4> tet_cs =
    {  1, 2, 3, 4,
      42,43,44,45  };

// ngon connectivities
  vector<I4> ngon_vtx =
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

  SUBCASE("find_boundary_vertices") {
    auto ngon_cs = std_e::view_as_vblock_range(ngon_vtx,ngon_eso);
    auto bnd_vertices = find_boundary_vertices(ngon_cs,parent_elts);

    vector<I4> expected_boundary_vertices = {
      10,11,12,13,
      5, 4, 3
    };

    CHECK( bnd_vertices == expected_boundary_vertices );
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
      std::move(ngon_vtx),
      6,9
    );
    emplace_child(ngons,new_DataArray("ElementStartOffset", std::move(ngon_eso)));
    emplace_child(ngons,new_DataArray("ParentElements", std::move(parent_elts)));

    tree_range zone_elements = {tris,tets,ngons};
    auto boundary_vertices = get_ordered_boundary_vertex_ids<I4>(zone_elements);

    vector<I4> expected_boundary_vertices = { 1, 2, 3, 4, 5, 10, 11, 12, 13, 42, 43, 44 };

    CHECK( boundary_vertices == expected_boundary_vertices );
  }
};
