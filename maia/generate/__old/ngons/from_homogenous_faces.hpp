#pragma once


#include "maia/connectivity/iter/connectivity.hpp"
#include "std_e/future/span.hpp"
#include "range/v3/view/transform.hpp"
#include "range/v3/view/concat.hpp"
#include "range/v3/view/single.hpp"
#include "range/v3/view/join.hpp"
#include "maia/utils/std_e_utils.hpp"


// Needed for connectivity<I,CK> to be considered a range of fixed size by range-v3
template<class I, class CK>
struct ranges::range_cardinality<connectivity<I,CK>>
  : std::integral_constant<ranges::cardinality, static_cast<ranges::cardinality>(CK::nb_nodes)> {};


template<class connectivity_range_type> auto
convert_to_interleaved_ngons(const connectivity_range_type& cs) {
  using connectivity_type = ranges::range_value_t<connectivity_range_type>;
  constexpr int N = connectivity_type::nb_nodes;
  auto interleaved_ngon = [](const auto& c){
    return ranges::views::concat(
      ranges::views::single(N),
      ranges::views::all(c)
    );
  };
  // As of march 2019, ranges::view::join_view.size() is lacking an overload in case the inner range is of fixed size.
  // The resulting join_view is then not a sized_view, which can be detrimental for performance
  return ranges::views::all(cs) | ranges::views::transform(interleaved_ngon) | ranges::views::join;
}


template<class connectivity_range_type> auto
convert_to_ngons(const connectivity_range_type& cs) {
  using connectivity_type = ranges::range_value_t<connectivity_range_type>;
  constexpr int N = connectivity_type::nb_nodes;
  // As of march 2019, ranges::view::join_view.size() is lacking an overload in case the inner range is of fixed size.
  // The resulting join_view is then not a sized_view, which can be detrimental for performance
  return std::make_pair(
    std_e::step_range(0,cs.size(),N),
    ranges::views::all(cs) | ranges::views::join
  );
}
