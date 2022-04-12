#if __cplusplus > 201703L
#include "maia/transform/flip_faces.hpp"

#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"
#include "maia/sids/element_sections.hpp"

using namespace cgns;


namespace maia {


template<ElementType_t face_type, class I> auto
flip_faces_of_type(std_e::span<I> connectivities) -> void {
  constexpr int n_vtx_of_face_type = number_of_vertices(face_type);
  auto cs = std_e::view_as_block_range<n_vtx_of_face_type>(connectivities);
  for (auto&& c : cs) {
    std::ranges::reverse(c);
  }
}


template<class I> auto
_flip_faces(tree& z) -> void {
  auto face_sections = surface_element_sections(z);
  for (tree& face_section: face_sections) {
    auto connectivities = ElementConnectivity<I>(face_section);
    auto face_type = element_type(face_section);
    switch(face_type){
      case TRI_3: {
        flip_faces_of_type<TRI_3>(connectivities);
        break;
      }
      case QUAD_4: {
        flip_faces_of_type<QUAD_4>(connectivities);
        break;
      }
      default: {
        throw std_e::not_implemented_exception("not implemented: flip_faces for ngon");
      }
    }
  }
}

auto
flip_faces(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");
  if (value(z).data_type()=="I4") return _flip_faces<I4>(z);
  if (value(z).data_type()=="I8") return _flip_faces<I8>(z);
  throw cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}


} // maia
#endif // C++>17
