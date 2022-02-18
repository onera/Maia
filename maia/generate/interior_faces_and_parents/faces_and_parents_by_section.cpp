#include "maia/generate/interior_faces_and_parents/faces_and_parents_by_section.hpp"


using namespace cgns;


namespace maia {


auto
_find_n_face_for_element_type(const auto& n_faces_by_type, ElementType_t elt_type) {
  auto it = std::ranges::find(cgns::all_face_types,elt_type);
  auto idx = it-begin(cgns::all_face_types);
  return n_faces_by_type[idx];
}

template<class I> auto
allocate_faces_and_parents_by_section(const std::array<I8,n_face_types>& n_faces_by_type) -> faces_and_parents_by_section<I> {
  faces_and_parents_by_section<I> fps;
  transform(
    fps,
    [&n_faces_by_type](auto& fp){
      I8 n_faces = _find_n_face_for_element_type(n_faces_by_type,fp.element_type);
      fp.resize(n_faces);
    }
  );
  return fps;
}


// Explicit instanciations of functions defined in this .cpp file
template auto allocate_faces_and_parents_by_section<I4>(const std::array<I8,n_face_types>& n_faces_by_type) -> faces_and_parents_by_section<I4>;
template auto allocate_faces_and_parents_by_section<I8>(const std::array<I8,n_face_types>& n_faces_by_type) -> faces_and_parents_by_section<I8>;

} // maia
