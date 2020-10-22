#pragma once


#include "range/v3/algorithm/copy.hpp"
#include "std_e/future/algorithm.hpp"
#include "cpp_cgns/array_utils.hpp"


namespace cgns {


template<class Range> auto
to_cgns_vector(Range&& rng, cgns::cgns_allocator& alloc) {
  using T = ranges::range_value_t<Range>;
  if constexpr (ranges::sized_range<Range>) { // possible to allocate once
    auto v = cgns::make_cgns_vector<T>(rng.size(),alloc);
    ranges::copy(FWD(rng),begin(v));
    return v;
  } else { // need to use a back_inserter
    auto v = cgns::make_cgns_vector<T>(alloc);
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
to_cgns_vector(cgns::cgns_allocator& alloc) {
  return ranges::make_pipeable([&alloc](auto&& rng) {
    return to_cgns_vector(FWD(rng),alloc);
  });
}


} // cgns
