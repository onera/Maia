#include "std_e/unit_test/doctest_pybind_mpi.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"
#include "maia/utils/mesh_dir.hpp"
#include "std_e/utils/file.hpp"
#include "cpp_cgns/tree_manip.hpp"

#include "maia/generate/interior_faces_and_parents/interior_faces_and_parents.hpp"


PYBIND_MPI_TEST_CASE("generate_interior_faces_and_parents",2) {
  int rk = std_e::rank(test_comm);
  std::string yaml_tree = std_e::file_to_string(maia::mesh_dir+"hex_2_prism_2_dist_"+std::to_string(rk)+".yaml");
  tree b = maia::to_node(yaml_tree);

  tree& z = cgns::get_node_by_name(b,"Zone");
  maia::generate_interior_faces_and_parents(z,test_comm);
  ELOG(z);
}
