#pragma once


namespace maia {


template<class I>
class poly_elt_t_reference {
  public:
  // ctor
    constexpr poly_elt_t_reference(I* offsets)
      : offsets(offsets)
    {}
  // reference semantics
    constexpr poly_elt_t_reference() = delete;
    constexpr poly_elt_t_reference(const poly_elt_t_reference&) = default;
    constexpr poly_elt_t_reference(poly_elt_t_reference&&) = default;

    constexpr poly_elt_t_reference& operator=(const poly_elt_t_reference& other) {
      I& this_elt_t_offset = *offsets;
      I& next_elt_t_offset = *(offsets+1);
      I new_elt_t_nb_of_nodes = other.elt_t_nb_of_nodes();
      next_elt_t_offset = this_elt_t_offset + new_elt_t_nb_of_nodes;
      return *this;
    }
    constexpr poly_elt_t_reference& operator=(poly_elt_t_reference&& other) {
      *this = other; // same as copy assignment
      return *this;
    }
    // operator= overloads for const types {
    template<class I0> auto
    // requires I0 is I or const I
    operator=(const poly_elt_t_reference<const I0>& other) -> decltype(auto) {
      I& this_elt_t_offset = *offsets;
      I& next_elt_t_offset = *(offsets+1);
      I new_elt_t_nb_of_nodes = other.elt_t_nb_of_nodes();
      next_elt_t_offset = this_elt_t_offset + new_elt_t_nb_of_nodes;
      return *this;
    }
    template<class I0> auto
    // requires I0 is I or const I
    operator=(poly_elt_t_reference<const I0>&& other) -> decltype(auto) {
      *this = other; // same as copy assignment
      return *this;
    }
    // }

  // conversion to an elt_t
    constexpr
    operator I() const {
      return elt_t_nb_of_nodes();
    }
    constexpr auto
    elt_t_nb_of_nodes() const -> I { // TODO RENAME (not nodes for NFace)
      I& this_elt_t_offset = *offsets;
      I& next_elt_t_offset = *(offsets+1);
      return next_elt_t_offset - this_elt_t_offset;
    }
  private:
    I* offsets;
};


} // maia
