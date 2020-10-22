#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices.hpp"

#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"

#include "maia/connectivity/iter_cgns/range.hpp"
#include "maia/utils/multi_array_utils.hpp"
#include "cpp_cgns/sids/utils.hpp"

#include "range/v3/view/zip.hpp"
#include <range/v3/action/sort.hpp>
#include <range/v3/action/unique.hpp>


namespace cgns {


auto
ngon_boundary_vertices(std_e::span<const I4> connectivities, md_array_view<const I4,2> parent_elts) -> std::vector<I4> {
  std::vector<I4> boundary_vertices;

  auto parent_elt_range = rows(parent_elts);
  auto connectivity_range = interleaved_ngon_range(connectivities);

  using namespace ranges::views;
  auto parent_and_connectivity_zip = zip(parent_elt_range,connectivity_range);

  for (const auto& [parent_elt,ngon] : parent_and_connectivity_zip ) {
    if (face_is_boundary(parent_elt)) {
      for (I4 vertex : ngon) {
        boundary_vertices.push_back(vertex);
      }
    }
  }

  return boundary_vertices;
}
        

auto
get_elements_boundary_vertices(const tree& elts) -> std::vector<I4> {
  // Preconditions:
  // - dimension == 3
  // - 2D elements are supposed to be on the boundary
  // - mixed elements are supposed to be volume elements only (Tet, Hex...)
  auto elt_type = ElementType<I4>(elts);
  auto connectivity = ElementConnectivity<I4>(elts);
  if (elt_type==cgns::NGON_n) {
    auto parent_elts = ParentElements<I4>(elts);
    return ngon_boundary_vertices(connectivity,parent_elts);
  } else if (std_e::contains(element_types_of_dimension(2),elt_type)) {
    return std::vector<I4>(connectivity.begin(),connectivity.end());
  } else {
    return {};
  }
}

auto
append_boundary_coordinates_indices(const tree& elts, std::vector<I4>& boundary_vertex_indices) -> void {
  auto elements_boundary_vertex_indices = get_elements_boundary_vertices(elts);
  std::copy(begin(elements_boundary_vertex_indices),end(elements_boundary_vertex_indices),std::back_inserter(boundary_vertex_indices));
}

auto
get_ordered_boundary_vertex_ids(const tree_range& elements_range) -> std::vector<I4> {
  /* Precondition: */ for ([[maybe_unused]] const tree& elts : elements_range) { STD_E_ASSERT(elts.label=="Elements_t"); }
  // Post-condition: the boundary nodes are unique and sorted

  std::vector<I4> boundary_vertex_ids;

  for(const tree& elts: elements_range) {
    append_boundary_coordinates_indices(elts,boundary_vertex_ids);
  }

  boundary_vertex_ids |= ranges::actions::sort | ranges::actions::unique;

  return boundary_vertex_ids;
}


} // cgns
