#if __cplusplus > 201703L
#include "maia/__old/transform/put_boundary_first/boundary_vertices.hpp"

#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"

#include "std_e/utils/concatenate.hpp"
#include "std_e/utils/vector.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"

#include "std_e/data_structure/block_range/block_range.hpp"
#include "std_e/data_structure/block_range/vblock_range.hpp"
#include "cpp_cgns/sids/utils.hpp"


using cgns::tree;
using cgns::tree_range;


namespace maia {


template<class I> auto
element_section_boundary_vertices(const tree& elt_section) -> std::vector<I> {
  STD_E_ASSERT(label(elt_section)=="Elements_t");

  auto elt_type = element_type(elt_section);
  STD_E_ASSERT(elt_type!=cgns::MIXED);
  // Other precondition: dimension == 3

  auto elt_vtx = cgns::ElementConnectivity<I>(elt_section);
  if (elt_type==cgns::NGON_n) {
    auto pe = cgns::ParentElements<I>(elt_section);
    auto eso = cgns::ElementStartOffset<I>(elt_section);
    auto cs = std_e::view_as_vblock_range(elt_vtx,eso);
    return find_boundary_vertices(cs,pe);
  } else if (cgns::element_dimension(elt_type)==2) {
    I n_bnd_elts = cgns::ElementSizeBoundary(elt_section);
    if (n_bnd_elts!=0) {
      I n_bnd_vertices = n_bnd_elts * cgns::number_of_vertices(elt_type);
      return std::vector<I>(begin(elt_vtx),begin(elt_vtx)+n_bnd_vertices);
    } else {
      // while it can be that the section is of mixed interior/exterior faces,
      // in all likelihood, all the elements are on the boundary
      // so we treat it so
      return std::vector<I>(begin(elt_vtx),end(elt_vtx));
      // TODO: pre-treat this case in Maia/CGNS conversion? (by fixing ElementSizeBoundary=n_elem in Maia)
    }
  } else { // 1D or 3D element
    return {};
  }
}

template<class I> auto
get_ordered_boundary_vertex_ids(const tree_range& element_sections) -> std::vector<I> {
  /* Precondition: */ for ([[maybe_unused]] const tree& e : element_sections) { STD_E_ASSERT(label(e)=="Elements_t"); }
  // Post-condition: the boundary nodes are unique and sorted

  std::vector<I> boundary_vertex_ids;

  for(const tree& elt_section: element_sections) {
    std_e::append(boundary_vertex_ids, element_section_boundary_vertices<I>(elt_section));
  }

  std_e::sort_unique(boundary_vertex_ids);

  return boundary_vertex_ids;
}


// Explicit instanciations of functions defined in this .cpp file
template auto get_ordered_boundary_vertex_ids<cgns::I4>(const tree_range& element_sections) -> std::vector<cgns::I4>;
template auto get_ordered_boundary_vertex_ids<cgns::I8>(const tree_range& element_sections) -> std::vector<cgns::I8>;



} // maia
#endif // C++>17
