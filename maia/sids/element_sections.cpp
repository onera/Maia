#include "maia/sids/element_sections.hpp"


using namespace cgns;


namespace maia {


auto
is_section_of_dimension(const tree& n, int dim) -> bool {
  return label(n)=="Elements_t" && element_dimension(element_type(n))==dim;
};

auto
elements_interval_of_dim(const tree& z, int dim) {
  auto is_section_of_dim = [dim](const tree& n) -> bool { return is_section_of_dimension(n,dim); };

  auto elts = get_children_by_predicate(z,is_section_of_dim);
  std::sort(begin(elts),end(elts),compare_by_range);

  if (!elts_ranges_are_contiguous(elts)) {
    throw cgns_exception("The ElementRange of Elements_t of dimension "+std::to_string(dim)+" are expected to be contiguous");
  }

  return
    interval<I8>(
      element_range(elts[0]).first(),
      element_range(elts.back()).last()
    );
}

auto
boundary_elements_interval(const tree& z) -> interval<I8> {
  return elements_interval_of_dim(z,2);
}
auto
volume_elements_interval(const tree& z) -> interval<I8> {
  return elements_interval_of_dim(z,3);
}


} // maia
