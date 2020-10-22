#pragma once


#include <algorithm>
#include "maia/connectivity/iter/connectivity_ref.hpp"

template<class I0, class I1, class Connectivity_kind>
class heterogenous_connectivity_ref {
  public:
  // traits
    using kind = Connectivity_kind;
    using index_type = std::remove_const_t<I1>;
    using elt_t_reference = typename kind::template elt_t_reference<I0>;

  // ctors
    heterogenous_connectivity_ref(elt_t_reference elt_t_ref, I1* nodes_ptr)
      : elt_t_ref(elt_t_ref)
      , nodes_ptr(nodes_ptr)
    {}

  // reference semantics
    heterogenous_connectivity_ref() = delete;
    heterogenous_connectivity_ref(const heterogenous_connectivity_ref&  other) = default;
    heterogenous_connectivity_ref(      heterogenous_connectivity_ref&& other) = default;

    heterogenous_connectivity_ref& operator=(const heterogenous_connectivity_ref& other) {
      std::copy(other.nodes_ptr, other.nodes_ptr+other.size(), nodes_ptr);
      elt_t_ref = other.elt_t_ref;
      return *this;
    }
    heterogenous_connectivity_ref& operator=(heterogenous_connectivity_ref&& other) {
      // even if the reference is temporary, we only care about the underlying values
      std::copy(other.nodes_ptr, other.nodes_ptr+other.size(), nodes_ptr);
      elt_t_ref = other.elt_t_ref;
      return *this;
    }
    // operator= overloads for const types {
    template<class,class,class> friend class heterogenous_connectivity_ref;
    template<class Integer0, class Integer1> auto
    // requires Integer0 is I0 or const I0
    // requires Integer1 is I1 or const I1
    operator=(const heterogenous_connectivity_ref<Integer0,Integer1,kind>& other) -> decltype(auto) {
      std::copy(other.nodes_ptr, other.nodes_ptr+other.size(), nodes_ptr);
      elt_t_ref = other.elt_t_ref;
      return *this;
    }
    template<class Integer0, class Integer1> auto
    // requires Integer0 is I0 or const I0
    // requires Integer1 is I1 or const I1
    operator=(heterogenous_connectivity_ref<Integer0,Integer1,kind>&& other) -> decltype(auto) {
      // even if the reference is temporary, we only care about the underlying values
      std::copy(other.nodes_ptr, other.nodes_ptr+other.size(), nodes_ptr);
      elt_t_ref = other.elt_t_ref;
      return *this;
    }
    // }

  // heterogenous accessors
    constexpr auto
    elt_t() const -> I1 {
      return elt_t_ref;
    }
    constexpr auto
    nb_nodes() const -> I1 {
      return kind::nb_nodes(elt_t());
    }

  // range interface
    constexpr auto size() const -> int { return nb_nodes(); }

    constexpr auto begin()       ->       I1* { return nodes_ptr; }
    constexpr auto begin() const -> const I1* { return nodes_ptr; }
    constexpr auto end()         ->       I1* { return nodes_ptr+size(); }
    constexpr auto end()   const -> const I1* { return nodes_ptr+size(); }

    template<class Integer> constexpr auto operator[](Integer i)       ->       I1& { return nodes_ptr[i]; }
    template<class Integer> constexpr auto operator[](Integer i) const -> const I1& { return nodes_ptr[i]; }
  private:
    elt_t_reference elt_t_ref;
    I1* nodes_ptr;
};
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator==(const heterogenous_connectivity_ref<I00,I01,CK>& x, const heterogenous_connectivity_ref<I10,I11,CK>& y) {
  if (x.elt_t() != y.elt_t()) return false;
  else return std::equal( x.begin() , x.end() , y.begin() );
}
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator!=(const heterogenous_connectivity_ref<I00,I01,CK>& x, const heterogenous_connectivity_ref<I10,I11,CK>& y) {
  return !(x == y);
}
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator< (const heterogenous_connectivity_ref<I00,I01,CK>& x, const heterogenous_connectivity_ref<I10,I11,CK>& y) {
  return
       x.elt_t()< y.elt_t()
   || (x.elt_t()==y.elt_t() && std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end()));
}

template<class I0, class I1, class CK> constexpr auto begin(const heterogenous_connectivity_ref<I0,I1,CK>& x) { return x.begin(); }
template<class I0, class I1, class CK> constexpr auto begin(      heterogenous_connectivity_ref<I0,I1,CK>& x) { return x.begin(); }
template<class I0, class I1, class CK> constexpr auto end  (const heterogenous_connectivity_ref<I0,I1,CK>& x) { return x.end(); }
template<class I0, class I1, class CK> constexpr auto end  (      heterogenous_connectivity_ref<I0,I1,CK>& x) { return x.end(); }

template<class I0, class I1, class CK> inline auto
to_string(const heterogenous_connectivity_ref<I0,I1,CK>& x) {
  return std_e::range_to_string(x);
}

