#pragma once


#include "cpp_cgns/cgnslib.h"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/sids/connectivity_category.hpp"
#include "maia/connectivity/iter/concepts.hpp"
#include "maia/connectivity/iter/poly_elt_t_kind.hpp"


namespace cgns {


template<int elt_type>
struct connectivity_kind {
  static constexpr int elt_t = elt_type;
  static constexpr int nb_nodes = cgns::number_of_nodes(elt_type);
};

struct mixed_kind {
  static constexpr int type = MIXED;
  static constexpr int nb_nodes(int n) { return cgns::number_of_nodes(n); }
  template<class I> using elt_t_reference = I&;
};

template<connectivity_category cat> struct connectivity_kind_of__impl;
template<> struct connectivity_kind_of__impl<            ngon > { using type = maia::indexed_polygon_kind; };
template<> struct connectivity_kind_of__impl<interleaved_ngon > { using type = maia::interleaved_polygon_kind; };
template<> struct connectivity_kind_of__impl<            nface> { using type = maia::indexed_polyhedron_kind; };
template<> struct connectivity_kind_of__impl<interleaved_nface> { using type = maia::interleaved_polyhedron_kind; };
template<> struct connectivity_kind_of__impl<            mixed> { using type = mixed_kind; };
template<> struct connectivity_kind_of__impl<interleaved_mixed> { using type = mixed_kind; };
template<connectivity_category cat> using connectivity_kind_of = typename connectivity_kind_of__impl<cat>::type;


} // cgns
