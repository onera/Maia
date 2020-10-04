#include <iostream>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// template<typename... Ts>
// using
// hvector = std::tuple<std::vector<Ts>...>;

int add(int i, int j) {
  return i + j;
}

template<typename T>
T
sub(T i, T j) {
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return i + j;
}

struct zone_structured {
  int global_id;
  std::string name;   //!< Name of the zone in the cgns_base

  // Ctor(s)
  zone_structured() = default;
  zone_structured(std::string name, int global_id) : name(name), global_id(global_id)
  {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }

  // Copy constructor(s)
  zone_structured (const zone_structured& rhs) = delete;
  zone_structured& operator=(const zone_structured &) = delete;

  // Move constructor(s)
  zone_structured (zone_structured&& other) = default;
  zone_structured& operator=(zone_structured&& rhs) = default;
};


// ---------------------------------------------------
struct zone_unstructured {
  int global_id;
  std::string name;   //!< Name of the zone in the cgns_base

  // Ctor(s)
  zone_unstructured() = default;
  zone_unstructured(std::string name, int global_id) : name(name), global_id(global_id)
  {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }

  // Copy constructor(s)
  zone_unstructured (const zone_unstructured& rhs) = delete;
  zone_unstructured& operator=(const zone_unstructured &) = delete;

  // Move constructor(s)
  zone_unstructured (zone_unstructured&& other) = default;
  zone_unstructured& operator=(zone_unstructured&& rhs) = default;

};

// ---------------------------------------------------
struct my_struct {
  int a;
  // std::tuple<zone_structured, zone_unstructured> h;
  zone_unstructured zones_u;
  my_struct(int a) : a(a), zones_u("un_nom", a)
  {
    std::cout << __PRETTY_FUNCTION__ << " --> " << &zones_u << std::endl;
    // auto& alpha = std::get<0>(h);
    // alpha = 4;
    // auto h = hvector<int, double>();
  }

  // Copy constructor(s)
  my_struct (const my_struct& rhs) = delete;
  my_struct& operator=(const my_struct &) = delete;

};


void test(zone_unstructured& zone){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
};

void test(zone_structured& zone){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
};

// ---------------------------------------------------
PYBIND11_MODULE(first_step, m) {
    m.doc() = "pybind11 first_step plugin"; // optional module docstring

    m.def("add", &add        , "A function which adds two numbers");
    m.def("sub", &sub<double>, "A function which sub two numbers");
    m.def("sub", &sub<int>   , "A function which sub two numbers");

    py::class_<my_struct>(m, "my_struct")
      .def(py::init<int>())
      .def_readwrite("a", &my_struct::a)
      .def_readonly("h", &my_struct::zones_u, py::return_value_policy::reference)
      // .def_readwrite("h", &my_struct::h)
      .def("__repr__", [](const my_struct& m){
        std::string s;
        s += "super_class " + std::to_string(m.zones_u.global_id) + "\n";
        return s;
      });

    py::class_<zone_unstructured> py_zone_unstructured(m, "zone_unstructured");
    py_zone_unstructured.def(py::init<std::string, int>())
      .def_readwrite("global_id", &zone_unstructured::global_id);

    py::class_<zone_structured> py_zone_structured(m, "zone_structured");
    py_zone_structured.def(py::init<std::string, int>())
      .def_readwrite("global_id", &zone_structured::global_id);

    m.def("lambda_t1", [](zone_unstructured& m){
      std::cout << __PRETTY_FUNCTION__ << &m << std::endl;
      return;
    });

    m.def("test", py::overload_cast<zone_structured&>(&test));
    m.def("test", py::overload_cast<zone_unstructured&>(&test));

}
