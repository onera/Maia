#if __cplusplus > 201703L
#include "std_e/unit_test/doctest_pybind.hpp"

#include "maia/utils/yaml/parse_yaml_cgns.hpp"

#include "cpp_cgns/tree_manip.hpp"

using namespace maia;
using namespace cgns;

PYBIND_TEST_CASE("parse_yaml_cgns") {
  SUBCASE("to_node") {
    std::string yaml_tree =
      "Base CGNSBase_t I4 [3,3]:\n"
      "  Zone0 Zone_t I4 [[ 8,1,0]]:\n"
      "  Zone1 Zone_t I4 [[12,2,0]]:\n";
    tree t = to_node(yaml_tree);

    CHECK(name (t) == "Base");
    CHECK(label(t) == "CGNSBase_t");
    CHECK(get_value<I4>(t) == std::vector{3,3});

    CHECK( name(get_child_by_name(t,"Zone0")) == "Zone0" );
    CHECK( name(get_child_by_name(t,"Zone1")) == "Zone1" );

    node_value& z0 = value(get_child_by_name(t,"Zone0"));
    CHECK( z0.rank() == 2 );
    CHECK( z0.extent(0) == 1 );
    CHECK( z0.extent(1) == 3 );
    CHECK( z0(0,0) == 8 );
    CHECK( z0(0,1) == 1 );
    CHECK( z0(0,2) == 0 );
  }

  SUBCASE("to_nodes") {
    std::string yaml_tree =
      "Zone0 Zone_t I4 [[ 8,1,0]]:\n"
      "Zone1 Zone_t I4 [[12,2,0]]:\n";
    std::vector<tree> t = to_nodes(yaml_tree);

    CHECK( t.size() == 2 );
    CHECK( name (t[0]) == "Zone0"  );
    CHECK( label(t[0]) == "Zone_t" );
    CHECK( children(t[0]).size() == 0 );
    CHECK( value(t[0]).rank() == 2 );
    CHECK( value(t[0]).extent(0) == 1 );
    CHECK( value(t[0]).extent(1) == 3 );
  }
}
#endif // C++>17
