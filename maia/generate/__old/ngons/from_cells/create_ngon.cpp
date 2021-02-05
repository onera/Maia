#include "maia/generate/__old/ngons/from_cells/create_ngon.hpp"

#include "maia/generate/__old/ngons/from_cells/faces_heterogenous_container.hpp"

#include "maia/utils/log/log.hpp"


namespace cgns {


auto
create_ngon(const faces_container<std::int32_t>& all_faces, std::int32_t first_ngon_id, factory& F) -> tree {
  using I = std::int32_t;
  auto _ = maia_time_log("create_ngon");
  // TODO explicit the fact that get<0> is tris and get<1> is quads
  const auto& bnd_faces_3 = std_e::get<0>(all_faces.boundary);
  const auto& bnd_faces_4 = std_e::get<1>(all_faces.boundary);
  const auto& int_faces_3 = std_e::get<0>(all_faces.interior);
  const auto& int_faces_4 = std_e::get<1>(all_faces.interior);
  std::cout << "size bnd_faces_3 = " << bnd_faces_3.size() << "\n";
  std::cout << "size bnd_faces_4 = " << bnd_faces_4.size() << "\n";
  std::cout << "size int_faces_3 = " << int_faces_3.size() << "\n";
  std::cout << "size int_faces_4 = " << int_faces_4.size() << "\n";
  I nb_ngons_boundary = bnd_faces_3.size()+bnd_faces_4.size();
  I nb_ngons_interior = int_faces_3.size()+int_faces_4.size();
  I nb_ngons = nb_ngons_boundary+nb_ngons_interior;
  auto ngon_cs = make_cgns_vector<I>(F.alloc()); // TODO reserve
  auto parent_elts = make_cgns_vector<I>(2*nb_ngons,F.alloc());
  I* l_parents = parent_elts.data();
  I* r_parents = parent_elts.data() + nb_ngons;
  I i=0;
  // boundary
  for (const auto& bnd_face_3 : bnd_faces_3) {
    ngon_cs.push_back(3);
    for (auto v : bnd_face_3.connec) {
      ngon_cs.push_back(v);
    }
    l_parents[i] = bnd_face_3.l_parent;
    r_parents[i] = 0;
    ++i;
  }
  for (const auto& bnd_face_4 : bnd_faces_4) {
    ngon_cs.push_back(4);
    for (auto v : bnd_face_4.connec) {
      ngon_cs.push_back(v);
    }
    l_parents[i] = bnd_face_4.l_parent;
    r_parents[i] = 0;
    ++i;
  }
  // interior
  for (const auto& int_face_3 : int_faces_3) {
    ngon_cs.push_back(3);
    for (auto v : int_face_3.connec) {
      ngon_cs.push_back(v);
    }
    l_parents[i] = int_face_3.l_parent;
    r_parents[i] = int_face_3.r_parent;
    ++i;
  }
  for (const auto& int_face_4 : int_faces_4) {
    ngon_cs.push_back(4);
    for (auto v : int_face_4.connec) {
      ngon_cs.push_back(v);
    }
    l_parents[i] = int_face_4.l_parent;
    r_parents[i] = int_face_4.r_parent;
    ++i;
  }

  tree ngons = F.newNgonElements(
    "Ngons",
    std_e::make_span(ngon_cs),
    first_ngon_id,first_ngon_id+nb_ngons-1,
    nb_ngons_boundary
  );
  emplace_child(ngons,F.new_DataArray("ParentElements", view_as_node_value(parent_elts)));
  return ngons;
}


} // cgns
