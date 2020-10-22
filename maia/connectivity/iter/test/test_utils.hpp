#pragma once

#include "maia/connectivity/iter/connectivity_range.hpp"

const int my_connectivity_type_id = 42;

struct my_connectivity_kind {
  static constexpr int elt_t = my_connectivity_type_id;
  static constexpr int nb_nodes = 3;
};

struct my_ngon_connectivity_kind {
  static auto nb_nodes(int type) {
    return type;
  }
  template<class I> using elt_t_reference = I&;
};
