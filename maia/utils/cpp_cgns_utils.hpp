#pragma once


#include "range/v3/algorithm/copy.hpp"
#include "range/v3/functional/pipeable.hpp"
#include "std_e/future/algorithm.hpp"
#include "std_e/buffer/buffer_vector.hpp"


namespace cgns {


template<class Range> auto
to_cgns_vector(Range&& rng) {
  using T = ranges::range_value_t<Range>;
  if constexpr (ranges::sized_range<Range>) { // possible to allocate once
    std_e::buffer_vector<T> v(rng.size());
    ranges::copy(FWD(rng),begin(v));
    return v;
  } else { // need to use a back_inserter
    auto v = std_e::make_buffer_vector<T>();
    std_e::copy(rng.begin(),rng.end(),std::back_inserter(v)); // ranges::copy does not compile (why?) and std::copy needs no sentinel
    return v;
  }
}

// If you are like me and don't know how this function work,
// don't panic! Its goal is just to provide
// an additional overload to the `to_cgns_vector` function above
// that is usable with the pipe (|) syntax.
// See also this talk by Eric Niebler: https://youtu.be/mFUXNMfaciE
constexpr auto
to_cgns_vector() {
  return ranges::make_pipeable([](auto&& rng) {
    return to_cgns_vector(FWD(rng));
  });
}


} // cgns
