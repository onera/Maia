#pragma once


#include <algorithm>
#include "maia/connectivity/iter/connectivity.hpp"


template<class I, class Connectivity_kind>
class connectivity_ref {
  public:
  // type traits
    using index_type = std::remove_const_t<I>;
    using kind = Connectivity_kind;
    static constexpr int N = kind::nb_nodes;
    static constexpr int nb_nodes = N;
    static constexpr int elt_t = kind::elt_t;

  // ctors
    connectivity_ref(I* ptr)
      : ptr(ptr)
    {}

  // reference semantics
    connectivity_ref() = delete;
    connectivity_ref(const connectivity_ref&  other) = default;
    connectivity_ref(      connectivity_ref&& other) = default;

    connectivity_ref& operator=(const connectivity_ref& other) {
      std::copy(other.ptr, other.ptr+other.size(), ptr);
      return *this;
    }

    connectivity_ref& operator=(const connectivity<I,kind>& other) {
      std::copy(other.data(), other.data()+other.size(), ptr);
      return *this;
    }

    connectivity_ref& operator=(const I(&other)[N]) {
      std::copy(other, other+N, ptr);
      return *this;
    }

    // operator= overloads for different const types {
    template<class I0,class CK> friend class connectivity_ref;
    template<class I0> auto
    // requires I0 is I or const I
    operator=(const connectivity_ref<I0,kind>& other) -> decltype(auto) {
      std::copy(other.ptr, other.ptr+other.size(), ptr);
      return *this;
    }
    // }

  // range interface
    static constexpr auto
    size() -> int {
      return nb_nodes;
    }

    constexpr auto data ()       ->       I* { return ptr; }
    constexpr auto data () const -> const I* { return ptr; }
    constexpr auto begin()       ->       I* { return ptr; }
    constexpr auto begin() const -> const I* { return ptr; }
    constexpr auto end  ()       ->       I* { return ptr + nb_nodes; }
    constexpr auto end  () const -> const I* { return ptr + nb_nodes; }

    template<class I0> constexpr auto operator[](I0 i)       ->       I& { return ptr[i]; }
    template<class I0> constexpr auto operator[](I0 i) const -> const I& { return ptr[i]; }
  private:
    I* ptr;
};

template<class I0, class I1, class CK> inline auto
operator==(const connectivity_ref<I0,CK>& x, const connectivity_ref<I1,CK>& y) {
  return std::equal( x.begin() , x.end() , y.begin() );
}
template<class I0, class I1, class CK> inline auto
operator!=(const connectivity_ref<I0,CK>& x, const connectivity_ref<I1,CK>& y) {
  return !(x == y);
}
template<class I0, class I1, class CK> inline auto
operator<(const connectivity_ref<I0,CK>& x, const connectivity_ref<I1,CK>& y) {
  return std::lexicographical_compare( x.begin() , x.end() , y.begin() , y.end() );
}
template<class I0, class I1, class CK> inline auto
swap(connectivity_ref<I0,CK> x, connectivity_ref<I1,CK> y) -> void {
  for (int i=0; i<x.size(); ++i) {
    using std::swap;
    swap(x[i],y[i]);
  }
}
template<class I, class CK> inline auto
to_string(const connectivity_ref<I,CK>& x) {
  return to_string(std_e::make_span(x.data(),x.size()));
}


template<class I, class CK> auto
begin(connectivity_ref<I,CK>& c) {
  return c.begin();
}
template<class I, class CK> auto
begin(const connectivity_ref<I,CK>& c) {
  return c.begin();
}
template<class I, class CK> auto
end(connectivity_ref<I,CK>& c) {
  return c.end();
}
template<class I, class CK> auto
end(const connectivity_ref<I,CK>& c) {
  return c.end();
}
