#pragma once


#include "maia/connectivity/iter_cgns/connectivity.hpp"


template<class I, class CK>
struct face_with_sorted_connectivity {
  connectivity<I,CK> connec;
  std::array<I,CK::nb_nodes> sorted_nodes;
  I l_parent;
};


template<class I, class CK> auto
operator==(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) -> bool {
  return 
      f0.connec == f1.connec
   && f0.sorted_nodes == f1.sorted_nodes
   && f0.l_parent == f1.l_parent;
}
template<class I, class CK> auto
operator!=(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) -> bool {
  return !(f0==f1);
}

template<class I, class CK> auto
same_face(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) -> bool {
  return f0.sorted_nodes == f1.sorted_nodes;
}
template<class I, class CK> auto
two_sides_of_same_face(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) -> bool {
  return
      same_face(f0,f1)
   && (!(f0.l_parent==f1.l_parent));
}

template<class I, class CK> auto
operator<(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) -> bool {
  return f0.sorted_nodes < f1.sorted_nodes;
}




template<class I, class CK>
struct interior_face {
  connectivity<I,CK> connec;
  I l_parent;
  I r_parent;
};
template<class I, class CK> auto
operator==(const interior_face<I,CK>& f0, const interior_face<I,CK>& f1) -> bool {
  return 
      f0.connec == f1.connec
   && f0.l_parent == f1.l_parent
   && f0.r_parent == f1.r_parent;
}
template<class I, class CK> auto
operator!=(const interior_face<I,CK>& f0, const interior_face<I,CK>& f1) -> bool {
    return !(f0==f1);
}

template<class I, class CK> auto
convert_to_interior_face(const face_with_sorted_connectivity<I,CK>& f0, const face_with_sorted_connectivity<I,CK>& f1) {
  // Precondition: two_sides_of_same_face(f0,f1)
  return interior_face<I,CK>{
    f0.connec,
    f0.l_parent,
    f1.l_parent
  };
}


template<class I, class CK>
struct boundary_face {
  connectivity<I,CK> connec;
  I l_parent;
};
template<class I, class CK> auto
operator==(const boundary_face<I,CK>& f0, const boundary_face<I,CK>& f1) -> bool {
  return 
      f0.connec == f1.connec
   && f0.l_parent == f1.l_parent;
}
template<class I, class CK> auto
operator!=(const boundary_face<I,CK>& f0, const boundary_face<I,CK>& f1) -> bool {
  return !(f0==f1);
}

template<class I, class CK> auto
convert_to_boundary_face(const face_with_sorted_connectivity<I,CK>& f0) {
  return boundary_face<I,CK>{
    f0.connec,
    f0.l_parent
  };
}


template<class I> using tri_3_with_sorted_connectivity = face_with_sorted_connectivity<I,cgns::connectivity_kind<cgns::TRI_3>>;
template<class I> using quad_4_with_sorted_connectivity = face_with_sorted_connectivity<I,cgns::connectivity_kind<cgns::QUAD_4>>;
template<class I> using interior_tri_3 = interior_face<I,cgns::connectivity_kind<cgns::TRI_3>>;
template<class I> using interior_quad_4 = interior_face<I,cgns::connectivity_kind<cgns::QUAD_4>>;
template<class I> using boundary_tri_3 = boundary_face<I,cgns::connectivity_kind<cgns::TRI_3>>;
template<class I> using boundary_quad_4 = boundary_face<I,cgns::connectivity_kind<cgns::QUAD_4>>;
