#include "std_e/unit_test/doctest.hpp"

#include "maia/generate/interior_faces_and_parents/element_faces_and_parents.hpp"
#include "maia/utils/yaml/parse_yaml_cgns.hpp"

using namespace cgns;
using namespace maia;
using std::vector;

TEST_CASE("generate_element_faces_and_parents") {
  std::string yaml_tree =
    "Quads Elements_t I4 [7,0]:\n"
    "  ElementRange IndexRange_t I4 [400,400]:\n"
    "  ElementConnectivity DataArray_t:\n"
    "    I4 : [ 40, 41, 42, 43 ]\n"
    "  :CGNS#Distribution UserDefinedData_t:\n"
    "    Element DataArray_t I8 [0, 1, 1]:\n"
    "Tris Elements_t I4 [5,0]:\n"
    "  ElementRange IndexRange_t I4 [300,301]:\n"
    "  ElementConnectivity DataArray_t:\n"
    "    I4 : [ 30, 31, 32,\n"
    "           37, 38, 39 ]\n"
    "  :CGNS#Distribution UserDefinedData_t:\n"
    "    Element DataArray_t I8 [0, 2, 2]:\n"
    "Hexas Elements_t I4 [17,0]:\n"
    "  ElementRange IndexRange_t I4 [800,810]:\n"
    "  ElementConnectivity DataArray_t:\n"
    "    I4 : [ 80, 81, 82, 83, 84, 85, 86, 87 ]\n"
    "  :CGNS#Distribution UserDefinedData_t:\n"
    "    Element DataArray_t I8 [2, 3, 10]:\n";
  vector<tree> elt_sections = maia::to_nodes(yaml_tree);
  cgns::tree_range elt_sections_rng(begin(elt_sections),end(elt_sections));

  faces_and_parents_by_section<I4> fps = generate_element_faces_and_parents<I4>(elt_sections_rng);
  CHECK( faces<TRI_3>(fps).size() == 2 ); // two tri faces
  CHECK( faces<TRI_3>(fps).connectivities()[0] == vector{30,31,32} );
  CHECK( faces<TRI_3>(fps).connectivities()[1] == vector{37,38,39} );
  CHECK( faces<TRI_3>(fps).parents() == vector{300,301} );
  CHECK( faces<TRI_3>(fps).parent_positions() == vector{1,1} );

  CHECK( faces<QUAD_4>(fps).size() == 1+6 ); // one quad face + 6 faces of 1 cube
  CHECK( faces<QUAD_4>(fps).connectivities()[0] == vector{40,41,42,43} );
  CHECK( faces<QUAD_4>(fps).connectivities()[1] == vector{80,83,82,81} );
  // ...
  CHECK( faces<QUAD_4>(fps).parents() == vector{400,802,802,802,802,802,802} ); // 800+2 because the distribution of node "Hexas" begins at 2
  CHECK( faces<QUAD_4>(fps).parent_positions() == vector{1,1,2,3,4,5,6} );
}
