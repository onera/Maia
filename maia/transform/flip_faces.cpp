#include "maia/transform/flip_faces.hpp"

#include "maia/connectivity/iter_cgns/range.hpp"
#include "maia/connectivity/iter_cgns/connectivity.hpp"
#include "cpp_cgns/sids/elements_utils.hpp"
#include "cpp_cgns/sids/utils.hpp"

using namespace cgns;

namespace maia {

auto
flip_faces(tree& b) -> void {
  for (tree& elt : get_nodes_by_matching(b,"Zone_t/Elements_t")) {
    if (element_dimension(element_type(elt))==2) {
      auto connectivity = ElementConnectivity<I4>(elt);
      if (element_type(elt)==TRI_3) {
        using con_range = connectivity_range<std_e::span<I4>,tri_3<I4>>;
        con_range cs(connectivity);
        for (auto&& c : cs) {
          std::reverse(begin(c),end(c));
        }
      } else if (element_type(elt)==QUAD_4) {
        using con_range = connectivity_range<std_e::span<I4>,quad_4<I4>>;
        con_range cs(connectivity);
        for (auto&& c : cs) {
          std::reverse(begin(c),end(c));
        }
      } else {
        throw std_e::not_implemented_exception("not implemented: flip_faces for ngon");
      }
    }
  }
}


} // maia
