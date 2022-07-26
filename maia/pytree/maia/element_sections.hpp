#pragma once

#include <algorithm>
#include "cpp_cgns/sids.hpp"

namespace maia {

template<class Tree> auto
element_sections(Tree& z) {
  return get_children_by_label(z,"Elements_t");
}

auto is_section_of_dimension(const cgns::tree& n, int dim) -> bool;
auto is_section_of_type(const cgns::tree& n, cgns::ElementType_t et) -> bool;

template<class Tree> auto
element_sections_of_dim(Tree& z, int dim) {
  auto is_section_of_dim = [dim](const cgns::tree& n) -> bool { return is_section_of_dimension(n,dim); };
  return get_children_by_predicate(z,is_section_of_dim);
}
template<class Tree> auto
surface_element_sections(Tree& z) {
  return element_sections_of_dim(z,2);
}
template<class Tree> auto
volume_element_sections(Tree& z) {
  return element_sections_of_dim(z,3);
}

template<class Tree> auto
element_sections_of_type(Tree& z, cgns::ElementType_t et) {
  STD_E_ASSERT(label(z)=="Zone_t");
  auto is_section_of_dim = [et](const cgns::tree& n) -> bool { return is_section_of_type(n,et); };
  return get_children_by_predicate(z,is_section_of_dim);
}
template<class Tree> auto
unique_element_section(Tree& z, cgns::ElementType_t et) {
  auto es_of_type = element_sections_of_type(z,et);
  STD_E_ASSERT(es_of_type.size()==1);
  return es_of_type[0];
}


template<class Tree> auto
element_sections_ordered_by_range(Tree& z) {
  auto elt_sections = element_sections(z);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  return elt_sections;
}
template<class Tree> auto
element_sections_of_dim_ordered_by_range(Tree& z, int dim) {
  auto elt_sections = element_sections_of_dim(z,dim);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  return elt_sections;
}
template<class Tree> auto
element_sections_ordered_by_range_by_type(Tree& z) {
  auto elt_sections = element_sections(z);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_elt_type);
  return elt_sections;
}
template<class Tree> auto
element_sections_ordered_by_range_type_dim(Tree& z) {
  auto elt_sections = element_sections(z);
  std::sort(begin(elt_sections),end(elt_sections),cgns::compare_by_range);
  std::stable_sort(begin(elt_sections),end(elt_sections),cgns::compare_by_elt_type_dim);
  return elt_sections;
}

auto max_element_id(const cgns::tree& z) -> cgns::I8;

template<class Tree_range> auto
elements_interval(const Tree_range& elt_sections) -> cgns::interval<cgns::I8> {
  using namespace cgns;
  STD_E_ASSERT(std::ranges::all_of(elt_sections,[](const auto& t){ return is_of_label(t,"Elements_t"); }));

  if (elt_sections.size()==0) { return interval<I8>(0,-1); }

  const_tree_range elts(begin(elt_sections),end(elt_sections)); // range of reference, so we can sort it locally
  std::ranges::sort(elts,compare_by_range);

  if (!elts_ranges_are_contiguous(elt_sections)) {
    std::string s;
    for (const auto& e : elt_sections) {
      s += name(e) + ',';
    }
    s.resize(s.size()-1);
    throw cgns_exception("The ElementRange of Elements_t \""+s+"\" are expected to be contiguous");
  }

  return
    interval<I8>(
      element_range(elts[0]).first(),
      element_range(elts.back()).last()
    );
}



} // maia
