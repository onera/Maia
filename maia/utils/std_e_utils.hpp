#pragma once


#include "range/v3/view/repeat_n.hpp"
#include "range/v3/view/exclusive_scan.hpp"


namespace std_e {


template<class I0, class I1, class I2> constexpr auto
step_range(I0 start, I1 n, I2 step) {
  return ranges::views::repeat_n(step,n) | ranges::views::exclusive_scan(start);
}


} // std_e
