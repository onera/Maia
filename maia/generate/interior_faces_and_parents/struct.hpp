#pragma once


#include <vector>
#include "cpp_cgns/sids/elements_utils.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"
#include "std_e/meta/transfom_values_to_tuple.hpp"


namespace maia {


// faces_and_parents_by_section {
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


//constexpr auto face_kinds = std_e::make_array(cgns::TRI_3,cgns::QUAD_4);
//
//
//template<class I>
//struct faces_and_parents_by_section {
//  template<ElementType_t et> using connectivities_with_parents_type = connectivities_with_parents<I,et>;
//  using impl_type = transform_values_to_tuple<face_kinds,connectivities_with_parents_type>;
//
//  faces_and_parents_by_section(I n_tri, I n_quad)
//    : tris(n_tri)
//    , quads(n_quad)
//  {}
//  connectivities_with_parents<I,cgns::TRI_3 > tris ;
//  connectivities_with_parents<I,cgns::QUAD_4> quads;
//};

template<class I>
struct faces_and_parents_by_section {
  faces_and_parents_by_section(I n_tri, I n_quad)
    : tris(n_tri)
    , quads(n_quad)
  {}
  connectivities_with_parents<I,cgns::TRI_3 > tris ;
  connectivities_with_parents<I,cgns::QUAD_4> quads;
};
// faces_and_parents_by_section }


// in_ext_faces_with_parents {
// TODO class because invariant: size is same for all (with connec by block)
// TODO add first_id
template<class I>
struct in_faces_with_parents {
  std::vector<I> connec;
  std::vector<I> l_parents;
  std::vector<I> r_parents;
  std::vector<I> l_parent_positions;
  std::vector<I> r_parent_positions;
  auto size() const -> I { return l_parents.size(); }
};
// TODO class because invariant: size is same for all (with connec by block)
// TODO add first_id
template<class I>
struct ext_faces_with_parents {
  std::vector<I> boundary_parents;
  std::vector<I> vol_parents;
  std::vector<I> vol_parent_positions;
  auto size() const -> I { return boundary_parents.size(); }
};
template<class I>
struct in_ext_faces_with_parents {
  in_faces_with_parents <I> in;
  ext_faces_with_parents<I> ext;
};
// in_ext_faces_with_parents }


} // maia
