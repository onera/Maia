#include "maia/transform/__old/partition_with_boundary_first/boundary_vertices.hpp"

#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

#include "std_e/algorithm/algorithm.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"

#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "maia/utils/multi_array_utils.hpp"
#include "cpp_cgns/sids/utils.hpp"

//#include "range/v3/view/zip.hpp"
#include <range/v3/action/sort.hpp>
#include <range/v3/action/unique.hpp>
#include "std_e/data_structure/multi_range/multi_range.hpp"


namespace cgns {


  // TODO use this
//auto
//parent_elements_range(const md_array_view<const I4,2>& pe) {
//  auto l_pe = std_e::column(pe,0);
//  auto r_pe = std_e::column(pe,1);
//  return std_e::view_as_multi_range2(l_pe,r_pe);
//}
//auto face_is_boundary2(const auto& pe) -> bool {
//  auto [l,r] = pe;
//  return l==0 || r==0;
//}
//
//
//auto
//ngon_boundary_vertices(std_e::span<const I4> connectivities, std_e::span<const I4> eso, md_array_view<const I4,2> parent_elts) -> std::vector<I4> {
//  std::vector<I4> boundary_vertices;
//
//  auto pe_rng = parent_elements_range(parent_elts);
//  auto connectivity_range = std_e::view_as_vblock_range(connectivities,eso);
//
//  auto parent_and_connectivity_zip = std_e::view_as_multi_range(pe_rng,connectivity_range);
//
//  for (const auto& [parent_elt,ngon] : parent_and_connectivity_zip) {
//    if (face_is_boundary2(parent_elt)) {
//      for (I4 vertex : ngon) {
//        boundary_vertices.push_back(vertex);
//      }
//    }
//  }
//
//  return boundary_vertices;
//}

// TMP
auto face_is_boundary3(const auto& l, const auto& r) -> bool {
  return l==0 || r==0;
}
auto
ngon_boundary_vertices(std_e::span<const I4> connectivities, std_e::span<const I4> eso, md_array_view<const I4,2> parent_elts) -> std::vector<I4> {
  std::vector<I4> boundary_vertices;

  auto l_pe = std_e::column(parent_elts,0);
  auto r_pe = std_e::column(parent_elts,1);
  auto connectivity_range = std_e::view_as_vblock_range(connectivities,eso);

  auto parent_and_connectivity_zip = std_e::view_as_multi_range(l_pe,r_pe,connectivity_range);

  for (const auto& [l_pe,r_pe,ngon] : parent_and_connectivity_zip) {
    if (face_is_boundary3(l_pe,r_pe)) {
      for (I4 vertex : ngon) {
        boundary_vertices.push_back(vertex);
      }
    }
  }

  return boundary_vertices;
}
// TMP end


auto
get_elements_boundary_vertices(const tree& elts) -> std::vector<I4> {
  // Preconditions:
  // - dimension == 3
  // - 2D elements are supposed to be on the boundary
  // - mixed elements are supposed to be volume elements only (Tet, Hex...)
  auto elt_type = element_type(elts);
  auto connectivity = ElementConnectivity<I4>(elts);
  auto eso = ElementStartOffset<I4>(elts);
  if (elt_type==cgns::NGON_n) {
    auto parent_elts = ParentElements<I4>(elts);
    return ngon_boundary_vertices(connectivity,eso,parent_elts);
  } else if (cgns::element_dimension(elt_type)==2) {
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
  /* Precondition: */ for ([[maybe_unused]] const tree& elts : elements_range) { STD_E_ASSERT(label(elts)=="Elements_t"); }
  // Post-condition: the boundary nodes are unique and sorted

  std::vector<I4> boundary_vertex_ids;

  for(const tree& elts: elements_range) {
    append_boundary_coordinates_indices(elts,boundary_vertex_ids);
  }

  boundary_vertex_ids |= ranges::actions::sort | ranges::actions::unique;

  return boundary_vertex_ids;
}


} // cgns
