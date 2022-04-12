#if __cplusplus > 201703L
#include "maia/sids/element_sections.hpp"


using namespace cgns;


namespace maia {


auto
is_section_of_dimension(const tree& n, int dim) -> bool {
  return label(n)=="Elements_t" && element_dimension(element_type(n))==dim;
};
auto
is_section_of_type(const tree& n, ElementType_t et) -> bool {
  return label(n)=="Elements_t" && element_type(n)==et;
};

auto
max_element_id(const tree& z) -> I8 {
  auto elt_sections = get_children_by_label(z,"Elements_t");
  STD_E_ASSERT(elt_sections.size()!=0);
  auto max_end_of_range = [](I8 acc, const tree& elt_sec){ return std::max(acc,element_range(elt_sec).last()); };
  return std::accumulate(begin(elt_sections),end(elt_sections),(I8)0,max_end_of_range);
}


} // maia
#endif // C++>17
