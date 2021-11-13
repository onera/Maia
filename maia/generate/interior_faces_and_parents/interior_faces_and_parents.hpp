#pragma once


#include "cpp_cgns/cgns.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "maia/connectivity/iter_cgns/connectivity_kind.hpp"
#include "maia/connectivity/iter/connectivity_range.hpp"
#include <mpi.h>


namespace maia {


// structs {
template<class I, cgns::ElementType_t elt_type>
class connectivities_with_parents {
  public:
    using connectivity_k = cgns::connectivity_kind<elt_type>;

    // Class invariant: connectivities().size() == parents().size()
    connectivities_with_parents(I n_connec)
      : connec(n_connec*number_of_nodes(elt_type))
      , parens(n_connec)
    {}

    auto
    size() {
      return parens.size();
    }

    auto
    connectivities() {
      return make_connectivity_range<connectivity_k>(connec);
    }

    auto
    parents() -> std_e::span<I> {
      return std_e::make_span(parens);
    }
  private:
    std::vector<I> connec;
    std::vector<I> parens;
};

template<class I>
struct faces_and_parents_t {
  faces_and_parents_t(I n_tri, I n_quad)
    : tris(n_tri)
    , quads(n_quad)
  {}
  connectivities_with_parents<I,cgns::TRI_3 > tris ;
  connectivities_with_parents<I,cgns::QUAD_4> quads;
};
// structs }




auto gen_interior_faces_and_parents(const cgns::tree_range& elt_sections) -> faces_and_parents_t<cgns::I4>; // TODO I8 also

auto generate_interior_faces_and_parents(cgns::tree& z, MPI_Comm comm) -> void;


} // maia
