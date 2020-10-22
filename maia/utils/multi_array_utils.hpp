#pragma once

#include "std_e/multi_array/multi_array.hpp"
#include "std_e/multi_array/multi_array/strided_array.hpp"
#include "range/v3/view/transform.hpp"
#include "range/v3/view/indices.hpp"


template<
  class M0, class M1,
  std::enable_if_t< M1::rank()==2 , int > =0
> constexpr auto
columns(std_e::multi_array<M0,M1>& x) {
  auto nb_cols = x.extent(1);
  auto col = [&x](auto i){ return std_e::column(x,i); };
  return ranges::views::indices(nb_cols) | ranges::views::transform(col);
}
template<
  class M0, class M1,
  std::enable_if_t< M1::rank()==2 , int > =0
> constexpr auto
columns(const std_e::multi_array<M0,M1>& x) {
  auto nb_cols = x.extent(1);
  auto col = [&x](auto i){ return std_e::column(x,i); };
  return ranges::views::indices(nb_cols) | ranges::views::transform(col);
}


template<
  class M0, class M1,
  std::enable_if_t< M1::rank()==2 , int > =0
> constexpr auto
rows(std_e::multi_array<M0,M1>& x) {
  auto nb_rows = x.extent(0);
  auto row = [&x](auto i){ return std_e::row(x,i); };
  return ranges::views::indices(nb_rows) | ranges::views::transform(row);
}
template<
  class M0, class M1,
  std::enable_if_t< M1::rank()==2 , int > =0
> constexpr auto
rows(const std_e::multi_array<M0,M1>& x) {
  auto nb_rows = x.extent(0);
  auto row = [&x](auto i){ return std_e::row(x,i); };
  return ranges::views::indices(nb_rows) | ranges::views::transform(row);
}
