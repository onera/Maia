#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"


namespace maia {


// structs {
template<class I, cgns::ElementType_t elt_type>
class connectivities_with_parents {
  public:
    static constexpr int n_vtx = number_of_vertices(elt_type);

    // Class invariant: connectivities().size() == parents().size() == parent_positions().size()
    connectivities_with_parents(I n_connec)
      : connec(n_connec*number_of_vertices(elt_type))
      , parens(n_connec)
      , par_pos(n_connec)
    {}

    auto
    size() {
      return parens.size();
    }

    auto
    connectivities() {
      return std_e::make_block_range<n_vtx>(connec);
    }

    auto
    parents() -> std_e::span<I> {
      return std_e::make_span(parens);
    }
    auto
    parent_positions() -> std_e::span<I> {
      return std_e::make_span(par_pos);
    }
  private:
    std::vector<I> connec;
    std::vector<I> parens;
    std::vector<I> par_pos;
};

template<class I>
struct faces_and_parents_by_section {
  faces_and_parents_by_section(I n_tri, I n_quad)
    : tris(n_tri)
    , quads(n_quad)
  {}
  connectivities_with_parents<I,cgns::TRI_3 > tris ;
  connectivities_with_parents<I,cgns::QUAD_4> quads;
};
// structs }




template<class I> auto
generate_element_faces_and_parents(const cgns::tree_range& elt_sections) -> faces_and_parents_by_section<I>;


} // maia
