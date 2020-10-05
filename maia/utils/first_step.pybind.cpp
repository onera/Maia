#include <iostream>
#include <tuple>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "std_e/data_structure/heterogenous_vector.hpp"

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
  std::string name;   //!< Name of the zone in the cgns_base
  int global_id;

  // Ctor(s)
  zone_structured() = default;
  zone_structured(std::string name, int global_id) : name(name), global_id(global_id)
  {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }
  ~zone_structured(){
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }

  // Copy constructor(s)
  zone_structured (const zone_structured& rhs) = delete;
  zone_structured& operator=(const zone_structured &) = delete;

  // Move constructor(s)
  zone_structured (zone_structured&& other) {
    std::cout << "oo" <<  __PRETTY_FUNCTION__ << std::endl;
    (*this) = std::move(other);
  }
  // zone_structured& operator=(zone_structured&& rhs) = default;
  zone_structured& operator=(zone_structured&& rhs) {
    rhs.global_id = -1;
    rhs.name      = "fake";
    global_id = -2;
    name      = "fake_move";
    std::cout <<  "aoo" <<__PRETTY_FUNCTION__ << std::endl;
    return *this;
  };
};


// ---------------------------------------------------
struct zone_unstructured {
  std::string name;   //!< Name of the zone in the cgns_base
  int global_id;

  // Ctor(s)
  zone_unstructured() = default;
  zone_unstructured(std::string name, int global_id) : name(name), global_id(global_id)
  {
    std::cout << __PRETTY_FUNCTION__ << std::endl;
  }
  ~zone_unstructured(){
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
struct cgns_base {
  std::string         name;
  int                 global_id;
  // std_e::hvector<zone_structured, zone_unstructured> zones;
  std::tuple<std::vector<zone_structured>, std::vector<zone_unstructured>> zones;

  // Ctor(s)
  cgns_base() = default;
  cgns_base(std::string name, int global_id)
    : name(name), global_id(global_id)
  {

  }
  // cgns_base(int global_id);

  // Copy constructor(s)
  cgns_base(const cgns_base& rhs) = delete;
  cgns_base& operator=(const cgns_base &) = delete;

  // Move constructor(s)
  cgns_base(cgns_base&& other) = default;
  cgns_base& operator=(cgns_base&& rhs) = default;
};


template<typename zone_type>
void add_zone_to_base(cgns_base& base, zone_type&& zone){
  auto& lzone = std::get<std::vector<zone_type>>(base.zones);
  std::cout << __PRETTY_FUNCTION__ << " --> " << zone.global_id << std::endl;
  lzone.push_back(std::move(zone));
  // auto tmp = std::move(zone);
};


// ---------------------------------------------------
struct my_struct {
  int a;
  // std::tuple<zone_structured, zone_unstructured> h;
  std::vector<zone_unstructured> h;
  zone_unstructured zones_u;
  my_struct(int a) : a(a), zones_u("un_nom", a)
  {
    h.push_back(std::move(zones_u));
    std::cout << __PRETTY_FUNCTION__ << " --> " << &zones_u << std::endl;
    // auto& alpha = std::get<0>(h);
    // alpha = 4;
    // auto h = hvector<int, double>();
  }

  // Copy constructor(s)
  my_struct (const my_struct& rhs) = delete;
  my_struct& operator=(const my_struct &) = delete;

};


void test(zone_unstructured& /*zone*/){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
};

void test(zone_structured& /*zone*/){
  std::cout << __PRETTY_FUNCTION__ << std::endl;
};

// struct essai {
//   std::string         name;
//   int                 global_id;
//   essai(std::string name, int global_id) : name(name), global_id(global_id)
//   {
//     std::cout << __PRETTY_FUNCTION__ << std::endl;
//   }
// };

class meboute {

  // operator meboute &&() && { return std::move(this); }
};

// struct custom {
//   static constexpr const char name[] = "Custom";
//   int foo() { return 1; }
// };

// namespace pybind11 { namespace detail {

// template < typename T >
// struct caster {
//     PYBIND11_TYPE_CASTER(T, T::name);
// };

// template <> struct type_caster< custom > : caster< custom > {};

// }} // namespace pybind11::detail

// PYBIND11_MODULE(core, m) {
//   py::class_< custom >(m, "cc")
//   .def("foo", &custom::foo);
// }


// PYBIND11_MAKE_OPAQUE(essai);

void comsume_ptr(std::tuple<int>&& a){
  auto tmp = std::move(a);
  std::cout << __PRETTY_FUNCTION__ << std::endl;
  return;
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
    .def_readonly("t", &my_struct::h, py::return_value_policy::automatic_reference)
    .def("__repr__", [](const my_struct& m){
      std::string s;
      s += "super_class " + std::to_string(m.zones_u.global_id) + "\n";
      return s;
    });

  py::class_<zone_unstructured> (m, "zone_unstructured")
    .def(py::init<std::string, int>())
    .def_readwrite("global_id", &zone_unstructured::global_id)
    .def_readwrite("name"     , &zone_unstructured::name     );

  py::class_<zone_structured> (m, "zone_structured")
    .def(py::init<std::string, int>())
    .def_readwrite("global_id", &zone_structured::global_id)
    .def_readwrite("name"     , &zone_structured::name     );

  // py::class_<essai> (m, "essai")
  //   .def(py::init<std::string, int>())
  //   .def_readwrite("global_id", &essai::global_id)
  //   .def_readwrite("name"     , &essai::name     );

  py::class_<cgns_base> (m, "cgns_base")
    .def(py::init<std::string, int>())
    .def_readwrite("global_id", &cgns_base::global_id)
    .def_readwrite("name"     , &cgns_base::name     )
    .def_readonly("zones"     , &cgns_base::zones, py::return_value_policy::automatic_reference)
    .def("__repr__", [](const cgns_base& m){
      std::string s;
      std_e::for_each(m.zones, [&](auto& zones) {
        std::for_each(begin(zones), end(zones), [&](auto& zone){
          s += "zone name : " + zone.name + " | " + std::to_string(zone.global_id);
          s += "\n";
        });
      });
      return s;
    });
  m.def("lambda_t1", [](zone_unstructured& m){
    std::cout << __PRETTY_FUNCTION__ << &m << std::endl;
    return;
  });

  m.def("test", py::overload_cast<zone_structured&>(&test));
  m.def("test", py::overload_cast<zone_unstructured&>(&test));

  // m.def("add_zone_to_base", &add_zone_to_base<zone_unstructured>, py::return_value_policy::automatic_reference);
  // m.def("add_zone_to_base", &add_zone_to_base<zone_structured>, py::return_value_policy::automatic_reference);

  // m.def("comsume_ptr", &comsume_ptr, py::return_value_policy::automatic_reference);


  // m.def("add_zone_to_base", [](zone_unstructured&& m){
  //     std::string s = "void";
  //     return s;
  //   });
  //   // m.def("add_zone_to_base", &add_zone_to_base<zone_structured>, py::return_value_policy::automatic_reference);

}
