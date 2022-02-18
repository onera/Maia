#pragma once


#include <vector>
#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/base/data_type.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"
#include "std_e/meta/transform_values_to_tuple.hpp"
#include "std_e/utils/tuple.hpp"


namespace maia {


template<class I, cgns::ElementType_t elt_type>
class connectivities_with_parents {
  public:
    static constexpr cgns::ElementType_t element_type = elt_type;
    static constexpr int n_vtx = number_of_vertices(elt_type);

    // Class invariant: connectivities().size() == parents().size() == parent_positions().size()
    connectivities_with_parents() = default;

    connectivities_with_parents(I n_connec)
      : connec(n_connec*number_of_vertices(elt_type))
      , parens(n_connec)
      , par_pos(n_connec)
    {}

    auto
    size() -> size_t {
      return parens.size();
    }
    auto
    resize(I n_connec) -> void {
      connec .resize(n_connec*number_of_vertices(elt_type));
      parens .resize(n_connec);
      par_pos.resize(n_connec);
    }

    auto
    connectivities() {
      return std_e::view_as_block_range<n_vtx>(connec);
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




// Wrapper around std::tuple< connectivities_with_parents<TRI_3>, ...<QUAD_4>, ...<TRI_6> ... >
template<class I>
class faces_and_parents_by_section {
  private:
    template<cgns::ElementType_t et> using connectivities_with_parents_type = connectivities_with_parents<I,et>;
    using impl_type = std_e::transform_values_to_tuple<cgns::all_face_types,connectivities_with_parents_type>;
    impl_type impl;

  public:
    faces_and_parents_by_section() = default;

    template<class I0, class F> friend auto
    transform(faces_and_parents_by_section<I0>& x, F f);

    template<cgns::ElementType_t face_type, class I0> friend auto
    faces(faces_and_parents_by_section<I0>& x) -> auto&;
};


template<class I> auto
allocate_faces_and_parents_by_section(const std::array<cgns::I8,cgns::n_face_types>& n_faces_by_type) -> faces_and_parents_by_section<I>;

template<class I, class F> auto
transform(faces_and_parents_by_section<I>& x, F f) {
  return std_e::transform(x.impl,f);
}

template<cgns::ElementType_t face_type, class I> auto
faces(faces_and_parents_by_section<I>& x) -> auto& {
  return std::get< connectivities_with_parents<I,face_type> >(x.impl);
}


} // maia
