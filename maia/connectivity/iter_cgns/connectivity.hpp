#pragma once

#include "maia/connectivity/iter/connectivity.hpp"
#include "maia/connectivity/iter_cgns/connectivity_kind.hpp"


namespace cgns {


template<class I, int elt_type> using cgns_connectivity = connectivity<I,connectivity_kind<elt_type>>; // TODO rename connectivity

template<class I> using tri_3   = cgns_connectivity<I,TRI_3  >;
template<class I> using quad_4  = cgns_connectivity<I,QUAD_4 >;

template<class I> using hex_8   = cgns_connectivity<I,HEXA_8 >;
template<class I> using tet_4   = cgns_connectivity<I,TETRA_4>;
template<class I> using penta_6 = cgns_connectivity<I,PENTA_6>;
template<class I> using pyra_5  = cgns_connectivity<I,PYRA_5 >;


} // cgns
