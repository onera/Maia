#pragma once


#include <type_traits>
#include "maia/connectivity/iter/connectivity_ref.hpp"
#include "std_e/meta/meta.hpp"


template<class I, class Connectivity_kind>
class connectivity_iterator {
  public:
    using index_type = std::remove_const_t<I>;
    using kind = Connectivity_kind;
    static constexpr int nb_nodes = kind::nb_nodes;

    /// std::iterator type traits
    using value_type = connectivity_ref<I,kind>; // TODO connectivity_view
    using reference = connectivity_ref<I,kind>;
    using difference_type = I;
    using iterator_category = std::forward_iterator_tag; // TODO random

    connectivity_iterator() = default;
    connectivity_iterator(I* ptr)
      : ptr(ptr)
    {}

    auto operator++() -> connectivity_iterator& {
      ptr += nb_nodes;
      return *this;
    }

    auto operator*() const {
      return reference(ptr);
    }

    auto data() const -> I* {
      return ptr;
    }

    friend constexpr auto
    operator==(const connectivity_iterator& it0, const connectivity_iterator& it1) {
      return it0.ptr == it1.ptr;
    }
    friend constexpr auto
    operator!=(const connectivity_iterator& it0, const connectivity_iterator& it1) {
      return !(it0 == it1);
    }
  private:
    I* ptr;
};
template<class I, class CK>
struct std::iterator_traits<connectivity_iterator<I,CK>> {
  using type = connectivity_iterator<I,CK>;
  using value_type = typename type::value_type;
  using reference = typename type::reference;
  using difference_type = typename type::difference_type;
  using iterator_category = typename type::iterator_category;
};


template<class C, class Connectivity_kind>
// requires C is a Contiguous_range
// requires I=C::value_type is an integer type
class connectivity_range {
  public:
    using I = std_e::add_other_type_constness<typename C::value_type,C>; // if the range is const, then make the content const

    using index_type = std::remove_const_t<I>;
    using kind = Connectivity_kind;
    static constexpr int nb_nodes = kind::nb_nodes;

    using iterator_type = connectivity_iterator<I,kind>;
    using const_iterator_type = connectivity_iterator<const I,kind>;
    using reference_type = connectivity_ref<I,kind>;
    using const_reference_type = connectivity_ref<const I,kind>;

    connectivity_range() = default;
    connectivity_range(C& cs)
      : cs_ptr(&cs)
    {}

    auto size() const -> std::ptrdiff_t {
      return cs_ptr->size();
    }

    auto begin()       ->       iterator_type { return {data()}; }
    auto begin() const -> const_iterator_type { return {data()}; }
    auto end()         ->       iterator_type { return {data() + size()}; }
    auto end()   const -> const_iterator_type { return {data() + size()}; }

    template<class Integer> auto operator[](Integer i)       ->       reference_type { return {data() + i*nb_nodes}; }
    template<class Integer> auto operator[](Integer i) const -> const_reference_type { return {data() + i*nb_nodes}; }

    auto push_back(reference_type c) {
      // requires C is a Container
      auto old_size = size();
      cs_ptr->resize( old_size + c.size() );

      auto c_position_in_cs = cs_ptr->begin() + old_size;
      std::copy(c.begin(),c.end(),c_position_in_cs);
    }

    auto data()       ->       I* { return cs_ptr->data(); }
    auto data() const -> const I* { return cs_ptr->data(); }
  private:
    C* cs_ptr;
};

template<class CK, class C> constexpr auto
make_connectivity_range(C& c) {
  return connectivity_range<std::remove_reference_t<C>,CK>(c);
}
