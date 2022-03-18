#include "maia/utils/cgns_tree_examples/unstructured_base.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "maia/generate/__old/from_structured_grid.hpp"
#include "maia/connectivity/iter/utility.hpp"
#include "range/v3/range/conversion.hpp"
#include "range/v3/view/repeat_n.hpp"
using namespace cgns;


// TODO move {
#include "range/v3/view/transform.hpp"
#include "range/v3/view/concat.hpp"
#include "range/v3/view/single.hpp"
#include "range/v3/view/join.hpp"
#include "maia/utils/std_e_utils.hpp"
template<class connectivity_range_type> auto
convert_to_ngons(const connectivity_range_type& cs) {
  using connectivity_type = ranges::range_value_t<connectivity_range_type>;
  constexpr int N = std::tuple_size_v<connectivity_type>;
  // As of march 2019, ranges::view::join_view.size() is lacking an overload in case the inner range is of fixed size.
  // The resulting join_view is then not a sized_view, which can be detrimental for performance
  return std::make_pair(
    std_e::step_range(0,cs.size(),N),
    ranges::views::all(cs) | ranges::views::join
  );
}
// END TODO }

auto
create_GridCoords0() -> tree {
  auto coord_X = node_value(
    { 0.,1.,2.,3.,
      0.,1.,2.,3.,
      0.,1.,2.,3.,
      0.,1.,2.,3.,
      0.,1.,2.,3.,
      0.,1.,2.,3. }
  );
  auto coord_Y = node_value(
    { 0.,0.,0.,0.,
      1.,1.,1.,1.,
      2.,2.,2.,2.,
      0.,0.,0.,0.,
      1.,1.,1.,1.,
      2.,2.,2.,2. }
  );
  auto coord_Z = node_value(
    { 0.,0.,0.,0.,
      0.,0.,0.,0.,
      0.,0.,0.,0.,
      1.,1.,1.,1.,
      1.,1.,1.,1.,
      1.,1.,1.,1. }
  );
  tree grid_coords = new_GridCoordinates();
  emplace_child(grid_coords,new_DataArray("CoordinateX",std::move(coord_X)));
  emplace_child(grid_coords,new_DataArray("CoordinateY",std::move(coord_Y)));
  emplace_child(grid_coords,new_DataArray("CoordinateZ",std::move(coord_Z)));
  return grid_coords;
}
auto
create_GridCoords1() -> tree {
  auto coord_X = node_value(
    { 3.,4.,
      3.,4.,
      3.,4.,
      3.,4. }
  );
  auto coord_Y = node_value(
    { 0.,0.,
      1.,1.,
      0.,0.,
      1.,1. }
  );
  auto coord_Z = node_value(
    { 0.,0.,
      0.,0.,
      1.,1.,
      1.,1. }
  );
  tree grid_coords = new_GridCoordinates();
  emplace_child(grid_coords,new_DataArray("CoordinateX",std::move(coord_X)));
  emplace_child(grid_coords,new_DataArray("CoordinateY",std::move(coord_Y)));
  emplace_child(grid_coords,new_DataArray("CoordinateZ",std::move(coord_Z)));
  return grid_coords;
}




auto
create_Zone0() -> tree {
/* Mesh used: "six quads", cf. simple_meshes.txt */
  int32_t VertexSize = 24;
  int32_t CellSize = 6;
  int32_t VertexSizeBoundary = 0;
  tree zone = new_UnstructuredZone("Zone0",{VertexSize,CellSize,VertexSizeBoundary});

  emplace_child(zone,create_GridCoords0());

  tree zone_bc = new_ZoneBC();
  emplace_child(zone_bc,new_BC("Inlet","FaceCenter",{1,2})); // 1,2 are the two i-faces at x=0
  emplace_child(zone,std::move(zone_bc));

  tree gc = new_GridConnectivity("MixingPlane","Zone1","FaceCenter","Abutting1to1");
  emplace_child(gc,new_PointList("PointList",{7})); // 7 is the bottom i-face at x=3
  emplace_child(gc,new_PointList("PointListDonor",{1})); // cf. zone 1
  tree zone_gc = new_ZoneGridConnectivity();
  emplace_child(zone_gc,std::move(gc));
  emplace_child(zone,std::move(zone_gc));


  std_e::multi_index<int32_t,3> vertex_dims = {4,3,2};
  //auto quad_faces = generate_faces(vertex_dims) | ranges::to<std::vector>;
  std::vector<std::array<int32_t,4>> quad_faces = generate_faces(vertex_dims) | ranges::to<std::vector>;
  auto [ngons_r,eso_r] = convert_to_ngons(quad_faces);
  auto ngons = ngons_r|ranges::to_vector;
  auto eso = eso_r|ranges::to_vector;
  std_e::offset(ngons,1); // CGNS is 1-indexed

  I8 nb_i_faces = 8;
  I8 nb_j_faces = 9;
  I8 nb_k_faces = 12;
  int32_t nb_ngons = nb_i_faces + nb_j_faces + nb_k_faces;

  auto i_faces_l_parent_elements = generate_faces_left_parent_cell_ids (vertex_dims,0);
  auto i_faces_r_parent_elements = generate_faces_right_parent_cell_ids(vertex_dims,0);
  auto j_faces_l_parent_elements = generate_faces_left_parent_cell_ids (vertex_dims,1);
  auto j_faces_r_parent_elements = generate_faces_right_parent_cell_ids(vertex_dims,1);
  // k-faces are considered interior (only for the sake of having interior nodes),
  // their parent is imaginary cell #42
  auto k_faces_l_parent_elements = ranges::views::repeat_n(42,nb_k_faces);
  auto k_faces_r_parent_elements = ranges::views::repeat_n(42,nb_k_faces);

  auto parent_elements = ranges::views::concat(
    i_faces_l_parent_elements , j_faces_l_parent_elements, k_faces_l_parent_elements,
    i_faces_r_parent_elements , j_faces_r_parent_elements, k_faces_r_parent_elements
  ) | ranges::to_vector;

  std_e::offset(parent_elements,1); // cgns indexing begin at 1
  std_e::multi_index<int,2> pe_dims = {(int)parent_elements.size()/2,2};
  md_array<int,2> parent_elts(std::move(parent_elements),pe_dims);

  tree ngon_elts = new_NgonElements(
    "Ngons",
    std::move(ngons),
    1,nb_ngons
  );
  emplace_child(ngon_elts,new_DataArray("ParentElements", node_value(std::move(parent_elts))));
  emplace_child(ngon_elts,new_DataArray("ElementStartOffset", node_value(std::move(eso))));
  emplace_child(zone,std::move(ngon_elts));

  return zone;
}

auto
create_Zone1() -> tree {
/* Le maillage utilisé est "one quad", cf. simple_meshes.h */
  int32_t VertexSize = 8;
  int32_t CellSize = 1;
  int32_t VertexSizeBoundary = 0;
  tree zone = new_UnstructuredZone("Zone1",{VertexSize,CellSize,VertexSizeBoundary});

  emplace_child(zone,create_GridCoords1());

  tree zone_bc = new_ZoneBC();
  emplace_child(zone_bc,new_BC("Outlet","FaceCenter",{2})); // 2 is the i-face at x=4
  emplace_child(zone,std::move(zone_bc));


  tree gc = new_GridConnectivity("MixingPlane","Zone0","FaceCenter","Abutting1to1");
  emplace_child(gc,new_PointList("PointList",{1})); // cf. zone 0
  emplace_child(gc,new_PointList("PointListDonor",{7})); // 1 is the i-face at x=3
  tree zone_gc = new_ZoneGridConnectivity();
  emplace_child(zone_gc,std::move(gc));
  emplace_child(zone,std::move(zone_gc));


  std_e::multi_index<int32_t,3> vertex_dims = {2,2,2};
  auto quad_faces = generate_faces(vertex_dims) | ranges::to_vector;
  auto [ngons_r,eso_r] = convert_to_ngons(quad_faces);
  auto ngons = ngons_r|ranges::to_vector;
  auto eso = eso_r|ranges::to_vector;
  std_e::offset(ngons,1); // CGNS is 1-indexed


  int32_t nb_ngons = 2 + 2 + 2;

  auto parent_elements = generate_faces_parent_cell_ids(vertex_dims) | ranges::to_vector;

  std_e::offset(parent_elements,1); // cgns indexing begin at 1
  std_e::multi_index<I8,2> pe_dims = {(I8)parent_elements.size()/2,2};
  md_array<int,2> parent_elts(std::move(parent_elements),std_e::dyn_shape<I8,2>(pe_dims));

  tree ngon_elts = new_NgonElements(
    "Ngons",
    std::move(ngons),
    1,nb_ngons
  );
  emplace_child(ngon_elts,new_DataArray("ParentElements", node_value(std::move(parent_elts))));
  emplace_child(ngon_elts,new_DataArray("ElementStartOffset", node_value(std::move(eso))));
  emplace_child(zone,std::move(ngon_elts));

  return zone;
}

auto
create_unstructured_base() -> cgns::tree {
  tree b = new_CGNSBase("Base0",3,3);
  emplace_child(b,create_Zone0());
  emplace_child(b,create_Zone1());
  return b;
}

