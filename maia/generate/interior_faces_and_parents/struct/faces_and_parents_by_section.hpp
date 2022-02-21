#pragma once


#include <vector>
#include "cpp_cgns/sids/elements_utils/faces.hpp"
#include "cpp_cgns/base/data_type.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"


namespace maia {


template<class I>
class connectivities_with_parents {
  // Class invariant: connectivities().size()/n_vtx_of_type == parents().size() == parent_positions().size()
  private:
    std::vector<I> connec;
    std::vector<I> parens;
    std::vector<I> par_pos;
  public:
  // ctors
    connectivities_with_parents() = default;

    connectivities_with_parents(cgns::ElementType_t elt_type, I n_connec)
      : connec(n_connec*number_of_vertices(elt_type))
      , parens(n_connec)
      , par_pos(n_connec)
    {}

  // Access
  //   Done through friend functions because one of them is template,
  //   and calling a template member function has an ugly syntax.
  //   The non-template other functions are also friend for consistency
    friend auto
    size(const connectivities_with_parents& x) -> size_t {
      return x.parens.size();
    }
    template<int n_vtx> friend auto
    connectivities(connectivities_with_parents& x) {
      return std_e::view_as_block_range<n_vtx>(x.connec);
    }
    friend auto
    parent_elements(connectivities_with_parents& x) -> std_e::span<I> {
      return std_e::make_span(x.parens);
    }
    friend auto
    parent_positions(connectivities_with_parents& x) -> std_e::span<I> {
      return std_e::make_span(x.par_pos);
    }
};

// TODO ?
template<class I>
using faces_and_parents_by_section = std::array< connectivities_with_parents<I> , cgns::n_face_types >;


template<class I> auto
allocate_faces_and_parents_by_section(const std::array<cgns::I8,cgns::n_face_types>& n_faces_by_type) -> faces_and_parents_by_section<I> {
  faces_and_parents_by_section<I> fps;
  for (int i=0; i<cgns::n_face_types; ++i) {
    cgns::ElementType_t face_type = cgns::all_face_types[i];
    cgns::I8 n_faces = n_faces_by_type[i];
    fps[i] = connectivities_with_parents<I>(face_type,n_faces);
  }
  return fps;
}


} // maia
