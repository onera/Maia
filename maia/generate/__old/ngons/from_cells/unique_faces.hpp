#pragma once

#include "std_e/algorithm/unique_compress.hpp"
#include "maia/generate/__old/ngons/from_cells/face_with_parents.hpp"
//#include "std_e/utils/time_logger.hpp"

#include <algorithm>
//#include <parallel/algorithm>


//namespace cgns {
//
//
//// TODO merge with what was done with unique_compress
//template<class I, class CK> auto
//append_unique_faces(
//    std::vector<face_with_sorted_connectivity<I,CK>>& faces,
//    std::vector<interior_face<I,CK>>& interior_faces
//) {
//    std_e::time_logger _("sort");
//    std::sort(faces.begin(),faces.end());
//    //__gnu_parallel::sort(faces.begin(),faces.end());
//
//    auto same_face_ = [](auto f0, auto f1){ return same_face(f0,f1); };
//    //auto convert_to_boundary_face_ = [](const auto& f0){ return convert_to_boundary_face(f0); };
//    auto get_boundary_face_ = [](const auto& f0){ return f0; };
//    auto convert_to_interior_face_ = [](const auto& f0, const auto& f1){ return convert_to_interior_face(f0,f1); };
//
//    std::vector<face_with_sorted_connectivity<I,CK>> boundary_faces;
//
//    //std_e::time_logger _2("merge_twin");
//    //std_e::merge_twin_copy(
//    //    faces.begin(),faces.end(),
//    //    std::back_inserter(boundary_faces),
//    //    std::back_inserter(interior_faces),
//    //    same_face_,
//    //    get_boundary_face_,
//    //    convert_to_interior_face_
//    //);
//
//    return boundary_faces;
//}
//
//
//} // cgns
//
//
#include "maia/generate/__old/ngons/from_cells/faces_heterogenous_container.hpp"
#include "maia/generate/__old/ngons/from_cells/generate_faces_from_connectivity_elts.hpp"
#include "std_e/base/not_implemented_exception.hpp"
//
//#include "std_e/utils/time_logger.hpp"
//
//
//namespace cgns {
//
//
//// TODO get<0> means get_tri and get<1> means get_quad
template<class I> auto
append_boundary_and_interior_faces(faces_container<I>& all_faces_unique, faces_heterogenous_container<I>& faces) {
  throw std_e::not_implemented_exception();
}
//  std_e::time_logger _("unique_faces");
//  // volume
//  auto& faces_3 = std_e::get<0>(faces.from_vol);
//  auto& all_interior_faces_3 = std_e::get<0>(all_faces_unique.interior);
//  auto& faces_4 = std_e::get<1>(faces.from_vol);
//  auto& all_interior_faces_4 = std_e::get<1>(all_faces_unique.interior);
//  std::vector<tri_3_with_sorted_connectivity<I>> bnd_faces_from_vol_3;
//  std::vector<quad_4_with_sorted_connectivity<I>> bnd_faces_from_vol_4;
//      bnd_faces_from_vol_3 = append_unique_faces(faces_3,all_interior_faces_3);
//      bnd_faces_from_vol_4 = append_unique_faces(faces_4,all_interior_faces_4);
//  //#pragma omp parallel sections
//  //{             
//  //  #pragma omp section
//  //  bnd_faces_from_vol_3 = append_unique_faces(faces_3,all_interior_faces_3);
//  //  #pragma omp section
//  //  bnd_faces_from_vol_4 = append_unique_faces(faces_4,all_interior_faces_4);
//  //}
//
//  // boundary
//  auto convert_to_boundary_face_ = [](const auto& f0){ return convert_to_boundary_face(f0); };
//  /// tris
//  auto& bnd_faces_from_face_3 = std_e::get<0>(faces.from_face);
//  std::sort(bnd_faces_from_face_3.begin(),bnd_faces_from_face_3.end());
//
//  auto& all_boundary_faces_3 = std_e::get<0>(all_faces_unique.boundary);
//  std::transform(
//    bnd_faces_from_face_3.begin(),bnd_faces_from_face_3.end(),
//    std::back_inserter(all_boundary_faces_3),
//    convert_to_boundary_face_
//  );
//
//  std::vector<tri_3_with_sorted_connectivity<I>> bnd_faces_not_from_face_3;
//  std::set_difference(
//    bnd_faces_from_vol_3.begin(),bnd_faces_from_vol_3.end(),
//    bnd_faces_from_face_3.begin(),bnd_faces_from_face_3.end(),
//    std::back_inserter(bnd_faces_not_from_face_3)
//  );
//  std::transform(
//    bnd_faces_not_from_face_3.begin(),bnd_faces_not_from_face_3.end(),
//    std::back_inserter(all_boundary_faces_3),
//    convert_to_boundary_face_
//  );
//
//  /// quads
//  auto& bnd_faces_from_face_4 = std_e::get<1>(faces.from_face);
//  std::sort(bnd_faces_from_face_4.begin(),bnd_faces_from_face_4.end());
//
//  auto& all_boundary_faces_4 = std_e::get<1>(all_faces_unique.boundary);
//  std::transform(
//    bnd_faces_from_face_4.begin(),bnd_faces_from_face_4.end(),
//    std::back_inserter(all_boundary_faces_4),
//    convert_to_boundary_face_
//  );
//
//  std::vector<quad_4_with_sorted_connectivity<I>> bnd_faces_not_from_face_4;
//  std::set_difference(
//    bnd_faces_from_vol_4.begin(),bnd_faces_from_vol_4.end(),
//    bnd_faces_from_face_4.begin(),bnd_faces_from_face_4.end(),
//    std::back_inserter(bnd_faces_not_from_face_4)
//  );
//  std::transform(
//    bnd_faces_not_from_face_4.begin(),bnd_faces_not_from_face_4.end(),
//    std::back_inserter(all_boundary_faces_4),
//    convert_to_boundary_face_
//  );
//}
//
//
//} // cgns
