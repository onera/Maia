#pragma once

#include "maia/connectivity/iter_cgns/connectivity_kind.hpp"

#include "maia/connectivity/iter/connectivity_range.hpp"
#include "maia/connectivity/iter/interleaved_connectivity_range.hpp"
#include "maia/connectivity/iter/indexed_poly_connectivity_range.hpp"
#include "maia/connectivity/iter/interleaved_connectivity_random_access_range.hpp"
#include "std_e/base/not_implemented_exception.hpp"
#include "cpp_cgns/sids/connectivity_category.hpp"


namespace cgns {


// interleaved (fwd and random access) {
template<class C> inline auto
interleaved_ngon_range(C& cs) {
  return make_interleaved_connectivity_range<maia::interleaved_polygon_kind>(cs);
}
template<class C> inline auto
interleaved_ngon_random_access_range(C& cs) {
  return make_interleaved_connectivity_random_access_range<maia::interleaved_polygon_kind>(cs);
}

template<class C> inline auto
interleaved_nface_range(C& cs) {
  return make_interleaved_connectivity_range<maia::interleaved_polyhedron_kind>(cs);
}
template<class C> inline auto
interleaved_nface_random_access_range(C& cs) {
  return make_interleaved_connectivity_random_access_range<maia::interleaved_polyhedron_kind>(cs);
}

template<class C> inline auto
interleaved_mixed_range(C& cs) {
  return make_interleaved_connectivity_range<mixed_kind>(cs);
}
template<class C> inline auto
interleaved_mixed_random_access_range(C& cs) {
  return make_interleaved_connectivity_random_access_range<mixed_kind>(cs);
}
// interleaved }


// indexed
template<class C0, class C1> inline auto
polygon_range(C0& offsets, C1& cs) {
  return maia::indexed_poly_connectivity_range<C0,C1,maia::indexed_polygon_kind>(offsets,cs);
}

template<class C0, class C1> inline auto
polyhedron_range(C0& offsets, C1& cs) {
  return maia::indexed_poly_connectivity_range<C0,C1,maia::indexed_polyhedron_kind>(offsets,cs);
}

template<class C0, class C1> inline auto
mixed_range(C0& offsets, C1& cs) {
  throw std_e::not_implemented_exception("indexed mixed_range");
  // TODO: something like this (not immediate because now interleaved_connectivity_random_access_range index array is self-hosted): 
  //return interleaved_connectivity_random_access_range<C0,C1,mixed_kind>(offsets,cs);
}
// indexed }



template<ElementType_t elt_type, class C> inline auto
fwd_connectivity_range(C& cs) {
       if constexpr (elt_type== NGON_n) { return  interleaved_ngon_range(cs); }
  else if constexpr (elt_type==NFACE_n) { return interleaved_nface_range(cs); }
  else if constexpr (elt_type== MIXED ) { return interleaved_mixed_range(cs); }
  else { return connectivity_range<C,connectivity_kind<elt_type>>(cs); }
}

template<class I, connectivity_category cat> inline auto
connectivity_vertex_range(tree& elt_pool) {
  auto cs = ElementConnectivity<I>(elt_pool);
  if constexpr (is_interleaved(cat)) {
    using connec_kind = connectivity_kind_of<cat>;
    return make_interleaved_connectivity_vertex_range<connec_kind>(cs);
  } else {
    return cs; // a non-interleaved connectivity range is composed of vertices only
  }
}


} // cgns
