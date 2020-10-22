#include "cpp_cgns/interop/pycgns_converter.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "cpp_cgns/node_manip.hpp"
#include "maia/transform/__old/partition_with_boundary_first/partition_with_boundary_first.hpp"
#include "maia/transform/__old/convert_to_simple_connectivities.hpp"
#include "maia/transform/__old/remove_ghost_info.hpp"
#include "maia/generate/__old/nfaces_from_ngons.hpp"

#include <Python.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static PyObject*
partition_with_boundary_first(PyObject*, PyObject* args) {
  PyObject* base_pytree;
  if (!PyArg_ParseTuple(args, "O", &base_pytree)) return NULL;
  cgns::tree base = cgns::view_as_cpptree(base_pytree);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::partition_with_boundary_first(base,cgns::factory(&alloc),MPI_COMM_WORLD);

  add_new_nodes_and_ownership(base,alloc,base_pytree);
  Py_INCREF(Py_None); return Py_None;
}

static PyObject*
sort_nface_into_simple_connectivities(PyObject*, PyObject* args) {
  PyObject* base_pytree;
  if (!PyArg_ParseTuple(args, "O", &base_pytree)) return NULL;
  cgns::tree base = cgns::view_as_cpptree(base_pytree);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::sort_nface_into_simple_connectivities(base,cgns::factory(&alloc));

  add_new_nodes_and_ownership(base,alloc,base_pytree);
  Py_INCREF(Py_None); return Py_None;
}

static PyObject*
convert_to_simple_connectivities(PyObject*, PyObject* args) {
  PyObject* base_pytree;
  if (!PyArg_ParseTuple(args, "O", &base_pytree)) return NULL;
  cgns::tree base = cgns::view_as_cpptree(base_pytree);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::convert_to_simple_connectivities(base,cgns::factory(&alloc));

  //add_new_nodes_and_ownership(base,alloc,base_pytree);
  update_and_transfer_ownership(base,alloc,base_pytree);
  Py_INCREF(Py_None); return Py_None;
}

static PyObject*
add_nfaces(PyObject*, PyObject* args) {
  PyObject* base_pytree;
  if (!PyArg_ParseTuple(args, "O", &base_pytree)) return NULL;
  cgns::tree base = cgns::view_as_cpptree(base_pytree);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::add_nfaces(base,cgns::factory(&alloc));

  add_new_nodes_and_ownership(base,alloc,base_pytree);
  Py_INCREF(Py_None); return Py_None;
}

static PyObject*
remove_ghost_info(PyObject*, PyObject* args) {
  PyObject* base_pytree;
  if (!PyArg_ParseTuple(args, "O", &base_pytree)) return NULL;
  cgns::tree base = cgns::view_as_cpptree(base_pytree);

  cgns::cgns_allocator alloc; // allocates and owns memory
  cgns::remove_ghost_info(base,cgns::factory(&alloc),MPI_COMM_WORLD);

  update_and_transfer_ownership2(base,alloc,base_pytree);
  Py_INCREF(Py_None); return Py_None;
}

PYBIND11_MODULE(maia_pybind, m) {
  m.doc() = "C++ maia functions wrapped by pybind";

  m.def("partition_with_boundary_first"        , &partition_with_boundary_first        , "ngon sorted with boundary faces first");
  m.def("sort_nface_into_simple_connectivities", &sort_nface_into_simple_connectivities, "sort nface into tet, prism, pyra, hex");
  m.def("convert_to_simple_connectivities"     , &convert_to_simple_connectivities     , "turn ngon with boundary first and nface to tri, quad, tet, prism, pyra, hex");
  m.def("add_nfaces"                           , &add_nfaces                           , "add nface Elements_t from ngons with ParentElements");
  m.def("remove_ghost_info"                    , &remove_ghost_info                    , "Remove ghost nodes and ghost elements of base");
}
