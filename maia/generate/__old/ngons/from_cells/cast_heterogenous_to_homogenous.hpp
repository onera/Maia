#pragma once


#include "maia/connectivity/iter_cgns/range.hpp"


namespace cgns {


template<ElementType_t ElementType, class I0, class I1, class Connectivity_kind> auto
cast_as(const heterogenous_connectivity_ref<I0,I1,Connectivity_kind>& het_con) {
  return connectivity_ref<const I1,connectivity_kind<ElementType>>(het_con.begin());
}


} // cgns
