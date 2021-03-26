#pragma once


#include "maia/connectivity/iter/interleaved_connectivity_range.hpp"


template<class I, class CK, class I_nc = std::remove_const_t<I>> auto
index_table(const interleaved_connectivity_range<I,CK>& cs_fwd_accessor) -> std::vector<I_nc> {
  std::vector<I_nc> idx_table = {0};
  I_nc idx = 0;
  for (const auto& c : cs_fwd_accessor) {
    idx += 1+c.nb_nodes();
    idx_table.push_back(idx);
  }
  return idx_table;
}


template<class I, class Connectivity_kind>
class interleaved_connectivity_random_access_iterator {
  public:
    using index_type = std::remove_const_t<I>;
    using kind = Connectivity_kind;

    using this_type = interleaved_connectivity_random_access_iterator<I,kind>;
    using fwd_it_type = interleaved_connectivity_iterator<I,kind>;

    using I_nc = std::remove_const_t<I>;

    /// std::iterator type traits
    using value_type = typename fwd_it_type::connec_view_type;
    using reference = typename fwd_it_type::connec_ref_type;
    using difference_type = I;
    using iterator_category = std::forward_iterator_tag; // TODO random

    interleaved_connectivity_random_access_iterator() = default;
    interleaved_connectivity_random_access_iterator(I* first, const std::vector<I_nc>& idx_table, I pos)
      : first(first)
      , pos(pos)
      , idx_table(idx_table)
    {}

    auto size() const -> I {
      return fwd_it()->size();
    }

    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0>
    auto operator+=(I0 i) -> this_type& {
      pos += i;
      return *this;
    }
    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0>
    auto operator-=(I0 i) -> this_type& {
      return *this += (-i);
    }
    auto operator++() -> this_type& { return (*this) += 1; }
    auto operator--() -> this_type& { return (*this) -= 1; }

    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0> friend auto
    operator+(const this_type& it, I0 i) -> this_type {
      this_type it0(it);
      it0 += i;
      return it0;
    }
    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0> friend auto
    operator-(const this_type& it, I0 i) -> this_type {
      this_type it0(it);
      it0 -= i;
      return it0;
    }
    friend auto
    operator-(const this_type& x, const this_type& y) -> I {
      STD_E_ASSERT(x.first==y.first); // iterators of same connectivity
      return x.pos-y.pos;
    }

    auto operator*() const {
      return *fwd_it();
    }
    auto operator->() const {
      return fwd_it().operator->();
    }

    friend inline auto
    operator==(const this_type& it0, const this_type& it1) -> bool {
      return it0.data()==it1.data();
    }
    friend inline auto
    operator!=(const this_type& it0, const this_type& it1) -> bool {
      return !(it0 == it1);
    }
    auto data() const -> I* {
      return first+idx_table[pos];
    }
  private:
  // member functions
    auto fwd_it() -> fwd_it_type {
      return {data()};
    }
    auto fwd_it() const -> const fwd_it_type {
      return {data()};
    }
  // data member
    I* first;
    I_nc pos;
    const std::vector<I_nc>& idx_table;
};
template<class I, class CK>
struct std::iterator_traits<interleaved_connectivity_random_access_iterator<I,CK>> {
  using type = interleaved_connectivity_random_access_iterator<I,CK>;
  using value_type = typename type::value_type;
  using reference = typename type::reference;
  using difference_type = typename type::difference_type;
  using iterator_category = typename type::iterator_category;
};


// interleaved_connectivity_random_access_range allows to random access connectivities
// by constructing an index table.
// WARNING: due to the heterogeneous structure of the connectivity collection being accessed,
// random access cannot and does not allow to replace a connectivity by another:
// only individual vertices can be mutated.
template<class I, class Connectivity_kind>
// requires I is an integer type
class interleaved_connectivity_random_access_range {
  public:
    using I_nc = std::remove_const_t<I>;

    using index_type = std::remove_const_t<I>;
    using kind = Connectivity_kind;

    using fwd_accessor_type = interleaved_connectivity_range<I,kind>;
    using value_type = typename fwd_accessor_type::value_type;
    using reference = typename fwd_accessor_type::reference;

    // TODO std::iterator traits
    using iterator = interleaved_connectivity_random_access_iterator<I,kind>;
    using const_iterator = interleaved_connectivity_random_access_iterator<const I,kind>;

    interleaved_connectivity_random_access_range() = default;
    interleaved_connectivity_random_access_range(std_e::span<I> cs)
      : fwd_accessor(cs)
      , idx_table(index_table(fwd_accessor))
    {}

    auto memory_length() const -> I {
      return fwd_accessor.memory_length();
    }
    auto size() const -> I {
      return idx_table.size()-1;
    }

    auto begin()       ->       iterator { return {data(), idx_table, 0     }; }
    auto begin() const -> const_iterator { return {data(), idx_table, 0     }; }
    auto end()         ->       iterator { return {data(), idx_table, size()}; }
    auto end()   const -> const_iterator { return {data(), idx_table, size()}; }

    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0>
    auto operator[](I0 i) -> reference {
      auto pos = data() + idx_table[i];
      auto& elt_t_ref = *pos;
      auto con_start = pos+1;
      return {elt_t_ref,con_start};
    }
    template<class I0, std::enable_if_t< std::is_integral_v<I0> , int > =0>
    auto operator[](I0 i) const -> const reference {
      auto pos = data() + idx_table[i];
      auto& elt_t_ref = *pos;
      auto con_start = pos+1;
      return {elt_t_ref,con_start};
    }

    auto push_back(const reference c) -> void {
      fwd_accessor.push_back(c);
      idx_table.push_back(idx_table.back() + c.memory_length);
    }

    auto data()       ->       I* { return fwd_accessor.data(); }
    auto data() const -> const I* { return fwd_accessor.data(); }
  private:
    fwd_accessor_type fwd_accessor;
    std::vector<I_nc> idx_table;
};

template<class CK, class I> constexpr auto
make_interleaved_connectivity_random_access_range(std_e::span<I> sp) {
  return interleaved_connectivity_random_access_range<I,CK>(sp);
}
template<class CK, class C> constexpr auto
make_interleaved_connectivity_random_access_range(C& c) {
  using I = std_e::add_other_type_constness<typename C::value_type,C>; // If the range is const, then make the content const
  std_e::span<I> sp(c.data(),c.size());
  return interleaved_connectivity_random_access_range<I,CK>(c);
}
