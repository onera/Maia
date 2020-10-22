#pragma once


/*
concept Connectivity_kind
  static constexpr int dimension
  static constexpr int nb_nodes
  static constexpr int order

concept Heterogenous_connectivity_kind
  static constexpr int dimension
  static constexpr nb_nodes(int type) -> int
  static constexpr order(int type) -> int
  template<class I> type_reference 


concept Element
  has kind is Connectivity_kind
  has index_type with is_integral(index_type)
  has vertex(int) -> index_type
  has vertices() -> connectivity<index_type,nb_vertices(Kind)>
*/
