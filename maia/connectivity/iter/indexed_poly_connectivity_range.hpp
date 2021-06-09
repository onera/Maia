#pragma once


#include <type_traits>
#include <algorithm>
#include "std_e/meta/meta.hpp"
#include "std_e/base/not_implemented_exception.hpp"
#include "maia/connectivity/iter/heterogenous_connectivity_ref.hpp"
#include "maia/connectivity/iter/poly_elt_t_kind.hpp"


namespace maia {


template<class I0, class I1, class Connectivity_kind>
class indexed_poly_connectivity_iterator { // TODO factor with std_e/iterator/index_iterator
  public:
  // type traits
    using index_type = std::remove_const_t<I1>;
    using kind = Connectivity_kind;

    using connec_view_type = heterogenous_connectivity_ref<I0,I1,kind>; // TODO heterogenous_connectivity_view
    using connec_ref_type = heterogenous_connectivity_ref<I0,I1,kind>;
       
    /// std::iterator type traits
    using value_type = connec_view_type;
    using reference = connec_ref_type;
    using difference_type = I1;
    using iterator_category = std::forward_iterator_tag; // TODO random

  // ctor
    indexed_poly_connectivity_iterator(){};

    indexed_poly_connectivity_iterator(I0* offsets_ptr, I1* cs_ptr)
      : offsets_ptr(offsets_ptr)
      , cs_ptr(cs_ptr)
    {}

  // iterator interface
    constexpr auto
    operator++() -> indexed_poly_connectivity_iterator& {
      ++offsets_ptr;
      return *this;
    }
    constexpr auto
    operator--() -> indexed_poly_connectivity_iterator& {
      --offsets_ptr;
      return *this;
    }
    template<class Integer> constexpr auto
    operator+=(Integer i) -> indexed_poly_connectivity_iterator& {
      offsets_ptr += i;
      return *this;
    }
    template<class Integer> constexpr auto
    operator-=(Integer i) -> indexed_poly_connectivity_iterator& {
      offsets_ptr -= i;
      return *this;
    }
    constexpr auto
    operator++(int) -> indexed_poly_connectivity_iterator {
      throw std_e::not_implemented_exception("don't use postfix operator++");
    }

    auto operator*() const -> reference { return {poly_elt_t_reference{offsets_ptr},cs_ptr+(*offsets_ptr)}; }

    template<class Integer>
    auto operator[](Integer i) const -> reference {
      auto offset_pos = offsets_ptr+i;
      auto connec_pos = cs_ptr+(*offset_pos);
      return {poly_elt_t_reference{offset_pos},connec_pos};
    }
    
    template<class I00, class I01, class I10, class I11, class CK> friend constexpr auto
    operator==(const indexed_poly_connectivity_iterator<I00,I01,CK>& x, const indexed_poly_connectivity_iterator<I10,I11,CK>& y) -> bool;
  private:
    I0* offsets_ptr;
    I1* cs_ptr;
};

template<class I00, class I01, class I10, class I11, class CK> constexpr auto
operator==(const indexed_poly_connectivity_iterator<I00,I01,CK>& x, const indexed_poly_connectivity_iterator<I10,I11,CK>& y) -> bool {
  return x.offsets_ptr==y.offsets_ptr;
}
template<class I00, class I01, class I10, class I11, class CK> constexpr auto
operator!=(const indexed_poly_connectivity_iterator<I00,I01,CK>& x, const indexed_poly_connectivity_iterator<I10,I11,CK>& y) -> bool {
  return !(x == y);
}
template<class I0, class I1, class Integer, class CK> constexpr auto
operator+(const indexed_poly_connectivity_iterator<I0,I1,CK>& x, Integer i) {
  indexed_poly_connectivity_iterator<I0,I1,CK> res(x);
  return res += i;
}
template<class I0, class I1, class Integer, class CK> constexpr auto
operator+(Integer i, const indexed_poly_connectivity_iterator<I0,I1,CK>& x) {
  return x+i;
}
template<class I0, class I1, class Integer, class CK> constexpr auto
operator-(const indexed_poly_connectivity_iterator<I0,I1,CK>& x, Integer i) {
  indexed_poly_connectivity_iterator<I0,I1,CK> res(x);
  return res -= i;
}
template<class I0, class I1, class Integer, class CK> constexpr auto
operator-(Integer i, const indexed_poly_connectivity_iterator<I0,I1,CK>& x) {
  return x-i;
}

} // maia
template<class I0, class I1, class CK>
struct std::iterator_traits<maia::indexed_poly_connectivity_iterator<I0,I1,CK>> {
  using type = maia::indexed_poly_connectivity_iterator<I0,I1,CK>;
  using value_type = typename type::value_type;
  using reference = typename type::reference;
  using difference_type = typename type::difference_type;
  using iterator_category = typename type::iterator_category;
};
namespace maia {


template<class C0, class C1, class Connectivity_kind>
// requires C0, C1 are Contiguous_range
// requires method C::data() returning ptr to first element
// requires I=C::value_type is an integer type
class indexed_poly_connectivity_range {
  public:
  // type traits
    using I0 = std_e::add_other_type_constness<typename C0::value_type,C0>; // If the range is const, then make the content const
    using I1 = std_e::add_other_type_constness<typename C1::value_type,C1>; // If the range is const, then make the content const
    using index_type = std::remove_const_t<I1>;
    using kind = Connectivity_kind;

    using iterator = indexed_poly_connectivity_iterator<I0,I1,kind>;
    using const_iterator = indexed_poly_connectivity_iterator<const I0,const I1,kind>;
    using reference = heterogenous_connectivity_ref<I0,I1,kind>;
    using const_reference = heterogenous_connectivity_ref<const I0,const I1,kind>;

  // ctors
    indexed_poly_connectivity_range() = default;

    indexed_poly_connectivity_range(C0& offsets, C1& cs)
      : offsets(&offsets)
      , cs(&cs)
    {}

  // accessors
    auto size() const -> index_type {
      return offsets->size()-1;
    }

    auto begin() -> iterator {
      return {
        offsets->data(),
        cs->data()
      };
    }
    auto begin() const -> const_iterator {
      return {
        offsets->data(),
        cs->data()
      };
    }
    auto end() -> iterator {
      auto offset_size = offsets->size()-1; // last position is the size of connectivities, not the offset
      return {
        offsets->data()+offset_size,
        nullptr // only the offset ptr is used for comparing iterators
      };
    }
    auto end() const -> const_iterator {
      auto offset_size = offsets->size()-1; // last position is the size of connectivities, not the offset
      return {
        offsets->data()+offset_size,
        nullptr // only the offset ptr is used for comparing iterators
      };
    }

    template<class Integer>
    auto operator[](Integer i) -> reference {
      return begin()[i];
    }
    template<class Integer>
    auto operator[](Integer i) const -> const_reference {
      return begin()[i];
    }

    template<class Reference>
    // requires Reference is reference or const_reference
    auto push_back(Reference c) -> void {
      // requires C0,C1 is a non-const Container
      auto last_offset = offsets->back();
      auto new_last_offset = last_offset + c.nb_nodes();
      offsets->push_back(new_last_offset);
      std::copy(c.begin(),c.end(),std::back_inserter(*cs));
    }

  private:
    C0* offsets;
    C1* cs;
};

template<class CK, class C0, class C1> constexpr auto
make_indexed_poly_connectivity_range(C0& offsets, C1& cs) {
  return indexed_poly_connectivity_range<C0,C1,CK>(offsets,cs);
}


} // maia
