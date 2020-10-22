#pragma once


#include <algorithm>
#include "maia/connectivity/iter/heterogenous_connectivity_ref.hpp"

template<class I0, class I1, class Connectivity_kind>
class heterogenous_connectivity_view {
  public:
  // traits
    using kind = Connectivity_kind;
    using index_type = std::remove_const_t<I1>;

  // ctors
    heterogenous_connectivity_view(I0 elt_type, I1* nodes_ptr)
      : elt_type(elt_type)
      , nodes_ptr(nodes_ptr)
    {}

    heterogenous_connectivity_view() = default;

    heterogenous_connectivity_view(const heterogenous_connectivity_view&  other) = default;
    heterogenous_connectivity_view(      heterogenous_connectivity_view&& other) = default;

    heterogenous_connectivity_view& operator=(const heterogenous_connectivity_view&  other) = default;
    heterogenous_connectivity_view& operator=(      heterogenous_connectivity_view&& other) = default;

    // from reference
    template<class I00, class I01>
    heterogenous_connectivity_view(const heterogenous_connectivity_ref<I00,I01,kind>& ref)
      : elt_type(ref.elt_t())
      , nodes_ptr(ref.begin())
    {}
    template<class I00, class I01>
    heterogenous_connectivity_view& operator=(const heterogenous_connectivity_ref<I00,I01,kind>& ref) {
      elt_type = ref.elt_t();
      nodes_ptr = ref.begin();
      return *this;
    }

    //// operator= overloads for const types {
    //template<class,class,class> friend class heterogenous_connectivity_view;
    //template<class Integer0, class Integer1> auto
    //// requires Integer0 is I0 or const I0
    //// requires Integer1 is I1 or const I1
    //operator=(const heterogenous_connectivity_view<Integer0,Integer1,kind>& other) -> decltype(auto) {
    //  return *this;
    //}
    //template<class Integer0, class Integer1> auto
    //// requires Integer0 is I0 or const I0
    //// requires Integer1 is I1 or const I1
    //operator=(heterogenous_connectivity_view<Integer0,Integer1,kind>&& other) -> decltype(auto) {
    //  // even if the reference is temporary, we only care about the underlying values
    //  return *this;
    //}
    //// }

  // heterogenous accessors
    constexpr auto
    elt_t() const -> I1 {
      return elt_type;
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
    I0 elt_type;
    I1* nodes_ptr;
};
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator==(const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
  if (x.elt_t() != y.elt_t()) return false;
  else return std::equal( x.begin() , x.end() , y.begin() );
}
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator!=(const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
  return !(x == y);
}
template<class I00, class I01, class I10, class I11, class CK> inline auto
operator< (const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
  return
       x.elt_t()< y.elt_t()
   || (x.elt_t()==y.elt_t() && std::lexicographical_compare(x.begin(),x.end(),y.begin(),y.end()));
}

template<class I0, class I1, class CK> constexpr auto begin(const heterogenous_connectivity_view<I0,I1,CK>& x) { return x.begin(); }
template<class I0, class I1, class CK> constexpr auto begin(      heterogenous_connectivity_view<I0,I1,CK>& x) { return x.begin(); }
template<class I0, class I1, class CK> constexpr auto end  (const heterogenous_connectivity_view<I0,I1,CK>& x) { return x.end(); }
template<class I0, class I1, class CK> constexpr auto end  (      heterogenous_connectivity_view<I0,I1,CK>& x) { return x.end(); }

template<class I0, class I1, class CK> inline auto
to_string(const heterogenous_connectivity_view<I0,I1,CK>& x) {
  return std_e::range_to_string(x);
}


//template<class I0, class I1, class Connectivity_kind>
//class heterogenous_connectivity_view {
//  public:
//  // traits
//    using kind = Connectivity_kind;
//    using index_type = std::remove_const_t<I1>;
//    using elt_t_reference = typename kind::template elt_t_reference<I0>;
//
//  // ctors
//    heterogenous_connectivity_view(elt_t_reference elt_t_ref, I1* nodes_ptr)
//    {}
//
//  // reference semantics
//    heterogenous_connectivity_view() = delete;
//    heterogenous_connectivity_view(const heterogenous_connectivity_view&  other) = default;
//    heterogenous_connectivity_view(      heterogenous_connectivity_view&& other) = default;
//
//    heterogenous_connectivity_view& operator=(const heterogenous_connectivity_view& other) {
//      return *this;
//    }
//    heterogenous_connectivity_view& operator=(heterogenous_connectivity_view&& other) {
//      // even if the reference is temporary, we only care about the underlying values
//      return *this;
//    }
//    template<class I00, class I01>
//    heterogenous_connectivity_view(const heterogenous_connectivity_ref<I00,I01,kind>& ref) {
//    }
//    template<class I00, class I01>
//    heterogenous_connectivity_view& operator=(const heterogenous_connectivity_ref<I00,I01,kind>& ref) {
//      return *this;
//    }
//
//    // operator= overloads for const types {
//    template<class,class,class> friend class heterogenous_connectivity_view;
//    template<class Integer0, class Integer1> auto
//    // requires Integer0 is I0 or const I0
//    // requires Integer1 is I1 or const I1
//    operator=(const heterogenous_connectivity_view<Integer0,Integer1,kind>& other) -> decltype(auto) {
//      return *this;
//    }
//    template<class Integer0, class Integer1> auto
//    // requires Integer0 is I0 or const I0
//    // requires Integer1 is I1 or const I1
//    operator=(heterogenous_connectivity_view<Integer0,Integer1,kind>&& other) -> decltype(auto) {
//      // even if the reference is temporary, we only care about the underlying values
//      return *this;
//    }
//    // }
//
//  // heterogenous accessors
//    constexpr auto
//    elt_t() const -> I1 {
//      return 0;
//    }
//    constexpr auto
//    nb_nodes() const -> I1 {
//      return kind::nb_nodes(elt_t());
//    }
//
//  // range interface
//    constexpr auto size() const -> int { return nb_nodes(); }
//
//    constexpr auto begin()       ->       I1* { return nullptr; }
//    constexpr auto begin() const -> const I1* { return nullptr; }
//    constexpr auto end()         ->       I1* { return nullptr+size(); }
//    constexpr auto end()   const -> const I1* { return nullptr+size(); }
//
//    template<class Integer> constexpr auto operator[](Integer i)       ->       I1& { throw ; }
//    template<class Integer> constexpr auto operator[](Integer i) const -> const I1& { throw ; }
//  private:
//};
//template<class I00, class I01, class I10, class I11, class CK> inline auto
//operator==(const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
//  throw;
//}
//template<class I00, class I01, class I10, class I11, class CK> inline auto
//operator!=(const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
//  return !(x == y);
//}
//template<class I00, class I01, class I10, class I11, class CK> inline auto
//operator< (const heterogenous_connectivity_view<I00,I01,CK>& x, const heterogenous_connectivity_view<I10,I11,CK>& y) {
//  throw;
//}
//
//template<class I0, class I1, class CK> constexpr auto begin(const heterogenous_connectivity_view<I0,I1,CK>& x) { return x.begin(); }
//template<class I0, class I1, class CK> constexpr auto begin(      heterogenous_connectivity_view<I0,I1,CK>& x) { return x.begin(); }
//template<class I0, class I1, class CK> constexpr auto end  (const heterogenous_connectivity_view<I0,I1,CK>& x) { return x.end(); }
//template<class I0, class I1, class CK> constexpr auto end  (      heterogenous_connectivity_view<I0,I1,CK>& x) { return x.end(); }
//
//template<class I0, class I1, class CK> inline auto
//to_string(const heterogenous_connectivity_view<I0,I1,CK>& x) {
//  return std_e::range_to_string(x);
//}
