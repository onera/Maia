#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "std_e/base/msg_exception.hpp"

namespace py = pybind11;

// Define accumulator functions for accumulate_by_stride
template<class T>
struct Reduce_Sum {
      static constexpr auto op = std::plus<>{};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = 0;
};
template <>
struct Reduce_Sum<bool> {
      static constexpr auto op = [] (const auto& x, const auto& y) {return int64_t(x) + int64_t(y);};
      using rtype = int64_t;
      static constexpr rtype neutral = 0;
};

template<class T>
struct Reduce_Prod {
      static constexpr auto op = std::multiplies<>{};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = 1;
};
template<class T>
struct Reduce_Max {
      static constexpr auto op = [](const T& x, const T& y) {return std::max(x, y);};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
};
template<class T>
struct Reduce_Min {
      static constexpr auto op = [](const T& x, const T& y) {return std::min(x, y);};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
};
template<class T>
struct Reduce_Land {
      static constexpr auto op = [](const T& x, const T& y) {return bool(x) && bool(y);};
      using rtype = bool; // Return type of op
      static constexpr rtype neutral = true;
};
template<class T>
struct Reduce_Lor {
      static constexpr auto op = [](const T& x, const T& y) {return bool(x) ||  bool(y);};
      using rtype = bool; // Return type of op
      static constexpr rtype neutral = false;
};
template<class T>
struct Reduce_Band {
      static constexpr auto op = [](const T& x, const T& y) {return x & y;};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = -1;
};
template <>
struct Reduce_Band<float> {
      static constexpr auto op = [] (float x, float y) {(void) x; (void)y; return 0.;}; // Fake OP
      using rtype = float;
      static constexpr rtype neutral = 0;
};
template <>
struct Reduce_Band<double> {
      static constexpr auto op = [] (double x, double y) {(void) x; (void)y; return 0.;}; // Fake OP
      using rtype = double;
      static constexpr rtype neutral = 0;
};

template<class T>
struct Reduce_Bor {
      static constexpr auto op = [](const T& x, const T& y) {return x | y;};
      using rtype = T; // Return type of op
      static constexpr rtype neutral = 0;
};
template <>
struct Reduce_Bor<float> {
      static constexpr auto op = [] (float x, float y) {(void) x; (void)y; return 0.;}; // Fake OP
      using rtype = float;
      static constexpr rtype neutral = 0;
};
template <>
struct Reduce_Bor<double> {
      static constexpr auto op = [] (double x, double y) {(void) x; (void)y; return 0.;}; // Fake OP
      using rtype = double;
      static constexpr rtype neutral = 0;
};

template<typename I, typename T>
void
sort_by_stride(const py::array_t<I> np_displs,
                     py::array_t<T> np_values) {

  auto n_elt   = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.mutable_data();
  
  for (ssize_t i=0; i<n_elt; ++i) {
    std::sort(values+displs[i], values+displs[i+1]);
  }
}


inline void _flip_by_stride_one(size_t n_elt, size_t item_size, std::byte* start, std::byte* end) {
  for (size_t j = 0; j < n_elt / 2; ++j) {
    auto left  = start + j*item_size;
    auto right = end - (j+1)*item_size;
    for (size_t k = 0; k < item_size; ++k) {
      std::swap(left[k], right[k]);
    }
  }
}

template<typename I>
void
flip_by_stride(py::array_t<I>&                   np_displs,
               py::array&                        np_values,
               std::optional<py::array_t<bool>>& np_mask) {

  size_t item_size = np_values.itemsize();

  auto displs = np_displs.data();
  auto start_ptr = static_cast<std::byte*>(np_values.mutable_data());

  if (np_mask.has_value()) {
    auto mask = np_mask.value().data();
    // Loop to operate on each section of the array
    for (ssize_t i=0; i < np_displs.size()-1; ++i) {
      if (mask[i]) {
        size_t n_elt = displs[i+1] - displs[i];
        auto start = start_ptr + displs[i]*item_size;
        auto end = start + n_elt*item_size;
        // Inverse subsection
        _flip_by_stride_one(n_elt, item_size, start, end);
      }
    }
  }
  else {
    // Loop to operate on each section of the array
    for (ssize_t i=0; i < np_displs.size()-1; ++i) {
      size_t n_elt = displs[i+1] - displs[i];
      auto start = start_ptr + displs[i]*item_size;
      auto end = start + n_elt*item_size;
      // Inverse subsection
      _flip_by_stride_one(n_elt, item_size, start, end);
    }
  }

}

template<typename I>
void
concatenate_by_stride(py::list       np_displs_l,
                      py::list       np_values_l,
                      py::array_t<I> np_displs_out, 
                      py::array      np_values_out) {


  int     n_input = np_values_l.size();
  int64_t n_elem  = np_displs_out.size() - 1;

  std::vector<const I*>          displs_ptrs(n_input);
  std::vector<const std::byte*>  values_ptrs(n_input);

  for (int j=0; j < n_input; ++j) {
    py::array_t<I> np_displs = np_displs_l[j].cast<py::array_t<I>>();
    py::array      np_values = np_values_l[j].cast<py::array>();
    displs_ptrs[j] = np_displs.data();
    values_ptrs[j] = static_cast<const std::byte*> (np_values.data());
  }

  auto displs_out = np_displs_out.mutable_data();
  auto write_buff = static_cast<std::byte*> (np_values_out.mutable_data());
  size_t s_data = np_values_out.request().itemsize;


  displs_out[0] = 0;
  for (int64_t i=0; i < n_elem; ++i) {

    displs_out[i+1] = 0;
    for (int j=0; j < n_input; ++j) {
      
      int count = displs_ptrs[j][i+1] - displs_ptrs[j][i];
      write_buff = std::copy_n(values_ptrs[j] + s_data*displs_ptrs[j][i],
                               count*s_data,
                               write_buff);

      displs_out[i+1] += displs_ptrs[j][i+1];
    }
  }
  
}

template<typename I, typename T>
void
roll_by_stride(py::array_t<I>& np_displs,
               py::array_t<T>& np_values,
               int             shift) {

  if (shift == 0) return;

  auto n_elt   = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.mutable_data();
  
  for (ssize_t i=0; i < n_elt; ++i) {
    auto count = displs[i+1] - displs[i];
    if (count > 0) {
      int loc_shift = shift % count;
      if (loc_shift < 0)
        loc_shift += count; // Always work with positive shift
      if (loc_shift == 0)
        continue;
      auto start = values + displs[i];
      auto middle = start + count - loc_shift;
      auto end = start + count;

      std::rotate(start, middle, end);
    
    }
  }
 
}


template<typename I, typename T>
std::tuple<py::array_t<I>, py::array_t<T>>
make_unique_by_stride(py::array_t<I>&   np_displs,
                      py::array_t<T>&   np_values) {

  int n_elt     = np_displs.size() - 1;
  auto displs   = np_displs.data();
  auto values   = np_values.data();
  
  auto np_displs_out = py::array_t<I>(n_elt+1); // To be returned
  auto displs_out    = np_displs_out.mutable_data();


  // Final array will be smaller
  T* array_out_tmp = new T[np_values.size()];

  displs_out[0] = 0;
  for (int i=0; i < n_elt; ++i) {
    int write_offset = 0;
    for (int j=displs[i]; j < displs[i+1]; ++j) {
      // Compare with k already written elts
      bool already_written = false;
      int k=0;
      while(k < write_offset && !already_written) {
        already_written = (array_out_tmp[displs_out[i] + k] == values[j]);
        k++;
      }
      if (!already_written) {
        array_out_tmp[displs_out[i] + write_offset++] = values[j];
      }
    }
    displs_out[i+1] = displs_out[i] + write_offset;
  }

  auto np_array_out = py::array_t<T>(displs_out[n_elt]); // To be returned
  memcpy(np_array_out.mutable_data(), array_out_tmp, displs_out[n_elt]*sizeof(T));

  delete[] array_out_tmp;
  return std::make_tuple(np_displs_out, np_array_out);
}



template<typename I, typename T, typename ReducOp>
py::array
_accumulate_by_stride(py::array_t<I>&   np_displs,
                      py::array_t<T>&   np_values,
                      ReducOp red) {

  auto n_elt   = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.data();

  using U = typename ReducOp::rtype;
  auto np_out = py::array_t<U>(n_elt);
  auto out    = np_out.mutable_data();
  
  for (ssize_t i=0; i < n_elt; ++i) {
    out[i] = std::accumulate(values+displs[i], values+displs[i+1], red.neutral, red.op);
  }

  return np_out;
}
template<typename I, typename T>
py::array
accumulate_by_stride(py::array_t<I>&   np_displs,
                     py::array_t<T>&   np_values,
                     const std::string& op)
{

  if (op == "SUM") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Sum<T>{});
  } else if (op == "PROD") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Prod<T>{});
  } else if (op == "MIN") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Min<T>{});
  } else if (op == "MAX") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Max<T>{});
  } else if (op == "LAND") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Land<T>{});
  } else if (op == "LOR") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Lor<T>{});
  } else if (op == "BAND") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Band<T>{});
  } else if (op == "BOR") {
      return _accumulate_by_stride(np_displs, np_values, Reduce_Bor<T>{});
  } else {
    throw std_e::msg_exception("Unvalid operation");
  }
}

template<typename I, typename T>
py::array_t<I>
indirect_outer_sort(py::array_t<I>&   np_displs,
                    py::array_t<T>&   np_values)
{
  auto n_elt  = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.data();

  // Array of indices (returned by function)
  py::array_t<I> np_perm(n_elt);
  auto perm = np_perm.mutable_data();

  // Initialise with 0..N-1 (arange)
  std::iota(perm, perm + n_elt, I{0});

  // Sort using custom comparison for lexicographic order
  std::stable_sort(perm, perm + n_elt,
    [displs, values](I a, I b) -> bool {
      I start_a = displs[a], end_a = displs[a+1];
      I start_b = displs[b], end_b = displs[b+1];

      auto len_a = end_a - start_a;
      auto len_b = end_b - start_b;
      auto minlen = std::min(len_a, len_b);

      const T* block_a = values + start_a;
      const T* block_b = values + start_b;

      for (I i = 0; i < minlen; ++i) {
        if (block_a[i] < block_b[i]) return true;
        if (block_b[i] < block_a[i]) return false;
      }
      return len_a < len_b;  // Common values are all equal => return shorter
    }
  );

  return np_perm;
}

template<typename I, typename T>
py::array_t<I>
indirect_outer_unique(py::array_t<I>&   np_displs,
                      py::array_t<T>&   np_values)
{
  auto n_elt  = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.data();

  // Indirect sort
  py::array_t<I> np_perm = indirect_outer_sort(np_displs, np_values);
  I* perm = np_perm.mutable_data();

  auto last = std::unique(perm, perm + n_elt,
    // This function return true if two blocks are equal (same size, same values)
    [displs, values](I a, I b) -> bool {
      I start_a = displs[a], end_a = displs[a+1];
      I start_b = displs[b], end_b = displs[b+1];
      auto len_a = end_a - start_a;
      auto len_b = end_b - start_b;
      if (len_a != len_b) return false;
      const T* block_a = values + start_a;
      const T* block_b = values + start_b;
      for (I i = 0; i < len_a; ++i) {
        if (block_a[i] != block_b[i]) return false;
      }
      return true;
    }
  );
   
  np_perm.resize({last-perm});
  return np_perm;
}

template<typename I1, typename I2>
void take(py::array_t<I1>      displs, 
          py::buffer           read_buff,
          py::array_t<I2>      ind, 
          py::buffer           write_buff)

{
  size_t s_data = read_buff.request().itemsize;
  std::byte* _read_buff  = static_cast<std::byte*> ( read_buff.request().ptr);
  std::byte* _write_buff = static_cast<std::byte*> (write_buff.request().ptr);

  auto _displs = displs.data();
  auto _ind    = ind.data();

  for (ssize_t i=0; i < ind.size(); ++i) {
    auto cur_idx = _ind[i];
    auto cur_cnt = _displs[cur_idx+1] - _displs[cur_idx];
    _write_buff = std::copy_n(_read_buff + s_data*_displs[cur_idx], 
                              cur_cnt*s_data,
                              _write_buff);
  }
}

template<typename I1, typename I2>
void
put(py::array_t<I1>  write_counts,
    py::buffer       write_buff,
    py::array_t<I2>  ind,
    py::array_t<I1>  read_counts,
    py::buffer       read_buff) {

  size_t s_data = write_buff.request().itemsize;

  auto _ind          = ind.data();
  auto _write_counts = write_counts.data();
  auto _read_counts  = read_counts.data();

  std::byte* _read_buff  = static_cast<std::byte*> ( read_buff.request().ptr);
  std::byte* _write_buff = static_cast<std::byte*> (write_buff.request().ptr);
  
  std::vector<I1> write_displs(write_counts.size()+1);
  write_displs[0] = 0;
  std::partial_sum(_write_counts, _write_counts+write_counts.size(), &write_displs[1]);

  int64_t r_idx = 0;
  for (int i=0; i < ind.size(); ++i) {
    auto idx = _ind[i];
    auto r_count = _read_counts[i];
    auto w_count = write_displs[idx+1] - write_displs[idx];
    auto w_start = write_displs[idx];

    // Write data **only** if r_count == w_count, otherwise we may erase other data 
    // or previously written data
    if (r_count == w_count) {
        std::copy_n(_read_buff + s_data*r_idx, 
                    r_count*s_data,
                    _write_buff + s_data*w_start);
    }
    r_idx += r_count;
  }
}

template<typename I1>
void
resize(py::array_t<I1>  write_counts,
       py::buffer       write_buff,
       py::array_t<I1>  read_counts,
       py::buffer       read_buff) {

  auto   n_data = write_counts.size();
  size_t s_data = write_buff.request().itemsize;

  auto _write_counts = write_counts.data();
  auto _read_counts  = read_counts.data();

  std::byte* _read_buff  = static_cast<std::byte*> ( read_buff.request().ptr);
  std::byte* _write_buff = static_cast<std::byte*> (write_buff.request().ptr);
  
  auto _cur_read  = _read_buff;
  auto _cur_write = _write_buff;
  for (int i=0; i < n_data; ++i) {
    auto r_count = _read_counts[i];
    auto w_count = _write_counts[i];

    auto count = std::min(r_count, w_count);
    std::copy_n(_cur_read, count*s_data, _cur_write);
   
    _cur_read += s_data*r_count;
    _cur_write += s_data*w_count;
  }
}

template<typename I1, typename I2>
void
put_extend(py::array_t<I1>  write_displs,
           py::buffer       write_buff,
           py::array_t<I2>  ind,
           py::array_t<I1>  read_counts,
           py::buffer       read_buff) {

  size_t s_data = write_buff.request().itemsize;

  auto _ind          = ind.data();
  auto _write_displs = write_displs.data();
  auto _read_counts  = read_counts.data();

  std::byte* _read_buff  = static_cast<std::byte*> ( read_buff.request().ptr);
  std::byte* _write_buff = static_cast<std::byte*> (write_buff.request().ptr);
  
  std::vector<I1> write_offset(write_displs.size(), 0);

  int64_t r_idx = 0;
  for (ssize_t i=0; i < ind.size(); ++i) {
    auto idx = _ind[i];
    auto r_count = _read_counts[i];
    auto w_start = _write_displs[idx] + write_offset[idx];

    std::copy_n(_read_buff + s_data*r_idx, 
                r_count*s_data,
                _write_buff + s_data*w_start);

    r_idx += r_count;
    write_offset[idx] += r_count;
  }
}


void register_vstride_module(py::module_& parent) {

  py::module_ m = parent.def_submodule("vstride");

  m.def("sort_by_stride", &sort_by_stride<int32_t, int32_t>, 
          py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int32_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int32_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int32_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());

  m.def("sort_by_stride", &sort_by_stride<int64_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int64_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int64_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("sort_by_stride", &sort_by_stride<int64_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());


  m.def("make_unique_by_stride", &make_unique_by_stride<int32_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int32_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int32_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int32_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());

  m.def("make_unique_by_stride", &make_unique_by_stride<int64_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int64_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int64_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("make_unique_by_stride", &make_unique_by_stride<int64_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert());

  m.def("roll_by_stride", &roll_by_stride<int32_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int32_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int32_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int32_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int32_t, bool>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());

  m.def("roll_by_stride", &roll_by_stride<int64_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int64_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int64_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int64_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());
  m.def("roll_by_stride", &roll_by_stride<int64_t, bool>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("shift").noconvert());

  m.def("accumulate_by_stride", &accumulate_by_stride<int32_t, bool>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int32_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int32_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int32_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int32_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int64_t, bool>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int64_t, int32_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int64_t, int64_t>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int64_t, float>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());
  m.def("accumulate_by_stride", &accumulate_by_stride<int64_t, double>, 
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("op").noconvert());



  m.def("flip_by_stride", &flip_by_stride<int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("mask").noconvert()=py::none());
  m.def("flip_by_stride", &flip_by_stride<int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(), py::arg("mask").noconvert()=py::none());

  m.def("indirect_outer_sort", &indirect_outer_sort<int32_t, bool>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int32_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int32_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int32_t, float>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int32_t, double>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int64_t, bool>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int64_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int64_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int64_t, float>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_sort", &indirect_outer_sort<int64_t, double>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());

  m.def("indirect_outer_unique", &indirect_outer_unique<int32_t, bool>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int32_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int32_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int32_t, float>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int32_t, double>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int64_t, bool>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int64_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int64_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int64_t, float>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("indirect_outer_unique", &indirect_outer_unique<int64_t, double>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());

  m.def("take", &take<int32_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(),
        py::arg("indices").noconvert(), py::arg("out").noconvert());
  m.def("take", &take<int32_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(),
        py::arg("indices").noconvert(), py::arg("out").noconvert());
  m.def("take", &take<int64_t, int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(),
        py::arg("indices").noconvert(), py::arg("out").noconvert());
  m.def("take", &take<int64_t, int64_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert(),
        py::arg("indices").noconvert(), py::arg("out").noconvert());

  m.def("put", &put<int32_t, int32_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put", &put<int32_t, int64_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put", &put<int32_t, int32_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put", &put<int64_t, int64_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());

  m.def("resize", &resize<int32_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("resize", &resize<int64_t>,
        py::arg("w_counts").noconvert(), py::arg("w_values").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());

  m.def("put_extend", &put_extend<int32_t, int32_t>,
        py::arg("w_displs").noconvert(), py::arg("w_values").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put_extend", &put_extend<int32_t, int64_t>,
        py::arg("w_values").noconvert(), py::arg("w_displs").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put_extend", &put_extend<int32_t, int32_t>,
        py::arg("w_values").noconvert(), py::arg("w_displs").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());
  m.def("put_extend", &put_extend<int64_t, int64_t>,
        py::arg("w_values").noconvert(), py::arg("w_displs").noconvert(), py::arg("indices").noconvert(),
        py::arg("r_counts").noconvert(), py::arg("r_values").noconvert());


  m.def("concatenate_by_stride", &concatenate_by_stride<int32_t>,
        py::arg("displs_l").noconvert(), py::arg("values_l").noconvert(),
        py::arg("displs_out").noconvert(), py::arg("values_out").noconvert());
  m.def("concatenate_by_stride", &concatenate_by_stride<int64_t>,
        py::arg("displs_l").noconvert(), py::arg("values_l").noconvert(),
        py::arg("displs_out").noconvert(), py::arg("values_out").noconvert());

}
