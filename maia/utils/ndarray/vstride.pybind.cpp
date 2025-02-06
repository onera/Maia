#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename I, typename T>
void
sort_by_stride(const py::array_t<I> np_displs,
                     py::array_t<T> np_values) {

  int n_elt   = np_displs.size() - 1;
  auto displs = np_displs.data();
  auto values = np_values.mutable_data();
  
  for (size_t i=0; i<n_elt; ++i) {
    std::sort(values+displs[i], values+displs[i+1]);
  }
}

template<typename I>
void
flip_by_stride(py::array_t<I>& np_displs,
               py::array     & np_values) {

  size_t item_size = np_values.itemsize();

  auto displs = np_displs.data();
  auto start_ptr = static_cast<std::byte*>(np_values.mutable_data());

  // Loop to operate on each section of the array
  for (size_t i=0; i < np_displs.size()-1; ++i) {
    size_t n_elt = displs[i+1] - displs[i];
    auto start = start_ptr + displs[i]*item_size;
    auto end = start + n_elt*item_size;

    // Inverse subsection
    for (size_t j = 0; j < n_elt / 2; ++j) {
      auto left  = start + j*item_size;
      auto right = end - (j+1)*item_size;
      for (size_t k = 0; k < item_size; ++k) {
          std::swap(left[k], right[k]);
      }
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

  for (size_t i=0; i < ind.size(); ++i) {
    auto cur_idx = _ind[i];
    auto cur_cnt = _displs[cur_idx+1] - _displs[cur_idx];
    std::copy_n(_read_buff + s_data*_displs[cur_idx], 
                cur_cnt*s_data,
                _write_buff);
    _write_buff += s_data*cur_cnt;
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


  m.def("flip_by_stride", &flip_by_stride<int32_t>,
        py::arg("displs").noconvert(), py::arg("values").noconvert());
  m.def("flip_by_stride", &flip_by_stride<int64_t>,
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

}
