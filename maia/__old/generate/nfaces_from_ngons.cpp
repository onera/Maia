#if __cplusplus > 201703L
#include "maia/__old/generate/nfaces_from_ngons.hpp"

#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/algorithm/for_each.hpp"


namespace cgns {


template<class I>
struct cell_id_and_face_id {
  I cell_id;
  I face_id;
};

constexpr auto equal_cell_id = [](const auto& x, const auto& y){ return x.cell_id == y.cell_id; };
constexpr auto less_cell_id = [](const auto& x, const auto& y){ return x.cell_id < y.cell_id; };

auto
face_ids_by_sorted_cell_ids(const tree& ngons) -> std::vector<cell_id_and_face_id<I4>> {
  STD_E_ASSERT(label(ngons)=="Elements_t");
  STD_E_ASSERT(element_type(ngons)==NGON_n);

  auto first_ngon_id = ElementRange<I4>(ngons)[0];
  auto parent_elts = ParentElements<I4>(ngons);

  std::vector<cell_id_and_face_id<I4>> cell_face_ids;
  for (I4 i=0; i<parent_elts.extent(0); ++i) {
    I4 ngon_id = i + first_ngon_id;
    I4 left_cell_id = parent_elts(i,0);
    I4 right_cell_id = parent_elts(i,1);
    cell_face_ids.push_back({left_cell_id,ngon_id});
    cell_face_ids.push_back({right_cell_id,ngon_id});
  }

  std::sort(begin(cell_face_ids),end(cell_face_ids),less_cell_id);
  return cell_face_ids;
}

template<class Forward_it, class Output_it> auto
append_nface_from_range(Forward_it f, Forward_it l, Output_it nfaces) {
  *nfaces++ = l-f;
  std::transform(f,l,nfaces,[](const auto& x){ return x.face_id; });
}

template<class Forward_it, class S, class Bin_predicate, class Range_function> constexpr auto
// requires Range_function(Forward_it,Forward_it)
// requires Predicate_generator is a Function_generator
// requires Predicate_generator::function_type(Forward_it::value_type) -> bool
for_each_partition2(Forward_it first, S last, Bin_predicate eq, Range_function f) {
  auto p_first = first;
  while (p_first!=last) {
    auto p_last = std_e::find_if(p_first,last,[=](auto& elt){ return !eq(*p_first,elt); });
    f(p_first,p_last);

    p_first = p_last;
  }
}
auto
nfaces_from_cell_face(const std::vector<cell_id_and_face_id<I4>>& cell_face_ids) -> tree {
  STD_E_ASSERT(std::is_sorted(begin(cell_face_ids),end(cell_face_ids),less_cell_id));

  auto first_non_boundary = std::partition_point(begin(cell_face_ids),end(cell_face_ids),[](auto& x){ return x.cell_id==0; });

  std::vector<I4> nfaces;
  I4 nb_nfaces = 0;

  auto append_nface = [&nb_nfaces,&nfaces](auto f, auto l){ append_nface_from_range(f,l,std::back_inserter(nfaces)); ++nb_nfaces; };
  for_each_partition2(first_non_boundary,end(cell_face_ids),equal_cell_id,append_nface);

  I4 first_nface_id = cell_face_ids[0].cell_id;
  return new_NfaceElements("Nface",std::move(nfaces),first_nface_id,first_nface_id+nb_nfaces);
}

auto
nfaces_from_ngons(const tree& ngons) -> tree {
  auto cell_face_ids = face_ids_by_sorted_cell_ids(ngons);
  return nfaces_from_cell_face(cell_face_ids);
}

auto
add_nfaces_to_zone(tree& z) -> void {
  tree& ngons = element_section(z,NGON_n);
  emplace_child(z,nfaces_from_ngons(ngons));
}

auto
add_nfaces(tree& b) -> void {
  for_each_unstructured_zone(b,add_nfaces_to_zone);
}


} // cgns
#endif // C++>17
