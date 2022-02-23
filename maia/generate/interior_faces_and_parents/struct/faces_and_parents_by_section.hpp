#pragma once


#include <vector>
#include "cpp_cgns/sids/elements_utils/faces.hpp"
#include "cpp_cgns/base/data_type.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"


namespace maia {


template<class I>
class connectivities_with_parents {
  // Class invariant: connectivities().size()/number_of_vertices(elt_type) == parents().size() == parent_positions().size()
  private:
    cgns::ElementType_t elt_type;
    std::vector<I> connec;
    std::vector<I> pe;
    std::vector<I> pp;
  public:
  // ctors
    connectivities_with_parents() = default;

    connectivities_with_parents(cgns::ElementType_t elt_type, I n_connec)
      : elt_type(elt_type)
      , connec(n_connec*number_of_vertices(elt_type))
      , pe(n_connec)
      , pp(n_connec)
    {}

  // Access
  //   Done through friend functions because one of them is template,
  //   and calling a template member function has an ugly syntax.
  //   The other functions are also friend for consistency
    friend auto
    size(const connectivities_with_parents& x) -> size_t {
      return x.pe.size();
    }
    friend auto
    element_type(const connectivities_with_parents& x) -> cgns::ElementType_t {
      return x.elt_type;
    }
    template<int n_vtx> friend auto
    connectivities(connectivities_with_parents& x) {
      STD_E_ASSERT(n_vtx == number_of_vertices(x.elt_type));
      return std_e::view_as_block_range<n_vtx>(x.connec);
    }
    friend auto
    parent_elements(connectivities_with_parents& x) -> std_e::span<I> {
      return std_e::make_span(x.pe);
    }
    friend auto
    parent_positions(connectivities_with_parents& x) -> std_e::span<I> {
      return std_e::make_span(x.pp);
    }
};


template<class I>
using faces_and_parents_by_section = std::vector<connectivities_with_parents<I>>;


} // maia
