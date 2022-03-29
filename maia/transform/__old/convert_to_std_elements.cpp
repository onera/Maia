#include "maia/transform/__old/convert_to_std_elements.hpp"


#include "maia/transform/__old/put_boundary_first/boundary_ngons_at_beginning.hpp" // TODO rename element_section_partition
#include "maia/connectivity/utils/connectivity_range.hpp"
#include "cpp_cgns/sids/Hierarchical_Structures.hpp"
#include "cpp_cgns/sids/Grid_Coordinates_Elements_and_Flow_Solution.hpp"
#include "std_e/future/span.hpp"
#include "cpp_cgns/sids/creation.hpp"
#include "cpp_cgns/sids/utils.hpp"
#include "std_e/data_structure/block_range/block_range.hpp"


using cgns::tree;
using cgns::I4;
using cgns::I8;


// TODO factor, nface/ngon iterator/ test
namespace maia {


template<class I> auto
make_ElementConnectivity_subrange(const cgns::tree& elt_section, I start, I finish) {
  auto cs      = cgns::ElementConnectivity<I>(elt_section);
  auto offsets = cgns::ElementStartOffset <I>(elt_section);

  return std_e::make_span(cs.data()+offsets[start],offsets[finish]-offsets[start]);
}
template<class I> auto
convert_to_simple_exterior_boundary_connectivities(const tree& ngons, I n_tri) -> std::pair<std::vector<tree>,I> {
  I n_face = ElementSizeBoundary(ngons);
  std::vector<tree> face_sections;

  auto tris  = make_ElementConnectivity_subrange(ngons, I(0),n_tri );
  if (tris.size() > 0) {
    face_sections.emplace_back(
      cgns::new_Elements(
        "TRI_3",
        cgns::TRI_3,
        std::vector(begin(tris),end(tris)),
        I(1),n_tri
      )
    );
  }

  auto quads = make_ElementConnectivity_subrange(ngons, n_tri,n_face);
  if (quads.size() > 0) {
    face_sections.emplace_back(
      cgns::new_Elements(
        "QUAD_4",
        cgns::QUAD_4,
        std::vector(begin(quads),end(quads)),
        n_tri+1,n_face
      )
    );
  }

  return { std::move(face_sections) , n_face+1 };
}

template<class T> auto
find_vertex_not_in_first(const T& connec_0, const T& connec_1) {
  for (auto vertex : connec_1) {
    if (std::find(begin(connec_0),end(connec_0),vertex)==end(connec_0)) return vertex;
  }
  STD_E_ASSERT(false); return connec_0[0];
}


template<class I> auto
convert_to_tet_vtx(const auto& tet_face, const tree& ngons, I first_cell_id) -> tree {
  auto face_vtx = make_connectivity_range<I>(ngons);
  auto first_ngon_id = ElementRange<I>(ngons)[0];
  auto pe = ParentElements<I>(ngons);

  I n_tet = tet_face.size();
  I n_vtx = n_tet*4;
  std::vector<I> tet_vtx(n_vtx);
  auto d_first = tet_vtx.data();

  for (I k=0; k<n_tet; ++k) {
    const auto& tet = tet_face[k];
    I tetra_id = first_cell_id+k;
    I tri_0_idx = tet[0]-first_ngon_id;
    I tri_1_idx = tet[1]-first_ngon_id;
    auto tri_0 = face_vtx[tri_0_idx];
    auto tri_1 = face_vtx[tri_1_idx];
    if (pe(tri_0_idx,0)==tetra_id) { // outward normal
      d_first = std::reverse_copy(begin(tri_0),end(tri_0),d_first);
    } else {
      STD_E_ASSERT(pe(tri_0_idx,1)==tetra_id); // inward normal
      d_first = std::copy(begin(tri_0),end(tri_0),d_first);
    }
    I other_vertex = find_vertex_not_in_first(tri_0,tri_1);
    *d_first++ = other_vertex;
  }

  return cgns::new_Elements(
    "TETRA_4",
    cgns::TETRA_4,
    std::move(tet_vtx),
    first_cell_id,first_cell_id+n_tet-1
  );
}
template<class I> auto
convert_to_pyra_vtx(const auto& pyra_face, const tree& ngons, I first_cell_id) -> tree {
  auto face_vtx = make_connectivity_range<I>(ngons);
  auto first_ngon_id = ElementRange<I>(ngons)[0];
  auto pe = ParentElements<I>(ngons);

  I n_pyra = pyra_face.size();
  I n_vtx = n_pyra*5;
  std::vector<I> pyra_vtx(n_vtx);
  auto d_first = pyra_vtx.data();

  for (I k=0; k<n_pyra; ++k) {
    const auto& pyra = pyra_face[k];
    I pyra_id = first_cell_id+k;
    for (int i=0; i<pyra.size(); ++i) {
      I face_idx = pyra[i]-first_ngon_id;
      auto face = face_vtx[face_idx];
      if (face.size()==4) {
        if (pe(face_idx,0)==pyra_id) { // outward normal
          d_first = std::reverse_copy(begin(face),end(face),d_first);
        } else {
          STD_E_ASSERT(pe(face_idx,1)==pyra_id); // inward normal
          d_first = std::copy(begin(face),end(face),d_first);
        }
        I tri_idx = pyra[(i+1)%5]-first_ngon_id; // since we just found the quad, face (i+1)%5 has to be a tri
        auto tri = face_vtx[tri_idx];
        I other_vertex = find_vertex_not_in_first(face,tri);
        *d_first++ = other_vertex;
        break;
      }
    }
  }

  return cgns::new_Elements(
    "PYRA_5",
    cgns::PYRA_5,
    std::move(pyra_vtx),
    first_cell_id,first_cell_id+n_pyra-1
  );
}


template<class I, class Node_connectivity> auto
contains_vertex(I vtx, const Node_connectivity& c) -> bool {
  return std::find(begin(c),end(c),vtx) != end(c);
}

template<class I, class Quad> auto
other_common(I vtx, const Quad& quad0, const Quad& quad1) -> I {
  std::array<I,4> q0;
  std::copy(begin(quad0),end(quad0),begin(q0));
  std::sort(begin(q0),end(q0));
  std::array<I,4> q1;
  std::copy(begin(quad1),end(quad1),begin(q1));
  std::sort(begin(q1),end(q1));
  std::array<I,2> q0_inter_q1;
  std::set_intersection(begin(q0),end(q0),begin(q1),end(q1),begin(q0_inter_q1));
  if (q0_inter_q1[0]!=vtx) return q0_inter_q1[0];
  else { STD_E_ASSERT(q0_inter_q1[0]==vtx); return q0_inter_q1[1]; }
}

template<class I, class T> auto
node_above(I vtx, const T& quads) -> I {
  auto contains_vtx = [vtx](const auto& quad){ return contains_vertex(vtx,quad); };
  auto quad0 = std::find_if(begin(quads),end(quads),contains_vtx);
  auto quad1 = std::find_if(quad0+1,end(quads),contains_vtx);
  STD_E_ASSERT(quad0!=end(quads));
  STD_E_ASSERT(quad1!=end(quads));
  return other_common(vtx,*quad0,*quad1);
}

template<class I> auto
convert_to_prism_vtx(const auto& prism_face, const tree& ngons, I first_cell_id) -> tree {
  auto face_vtx = make_connectivity_range<I>(ngons);
  auto first_ngon_id = ElementRange<I>(ngons)[0];
  auto pe = ParentElements<I>(ngons);

  I n_prism = prism_face.size();
  I n_vtx = n_prism*6;
  std::vector<I> prism_vtx(n_vtx);
  auto d_first = prism_vtx.data();

  for (I k=0; k<n_prism; ++k) {
    const auto& prism = prism_face[k];
    I prism_id = first_cell_id+k;
    const I not_found = -1;
    I tri_idx = not_found;
    std::array<I,3> tri;
    std::array<std_e::array<I,4>,3> quads;
    int quad_pos = 0;
    for (const auto& face_id : prism) {
      I face_idx = face_id-first_ngon_id;
      auto face = face_vtx[face_idx];
      if (face.size()==3) {
        if (tri_idx==not_found) {
          tri_idx = face_idx;
          std::copy(begin(face),end(face),begin(tri));
        }
      } else {
        quads[quad_pos] = face;
        ++quad_pos;
      }
    }
    // use the tri as nodes 1,2,3
    if (pe(tri_idx,0)==prism_id) { // outward normal
      std::reverse(begin(tri),end(tri));
    } else {
      STD_E_ASSERT(pe(tri_idx,1)==prism_id);
    }
    d_first = std::copy(begin(tri),end(tri),d_first);

    // nodes "above" tri
    *d_first++ = node_above(tri[0],quads);
    *d_first++ = node_above(tri[1],quads);
    *d_first++ = node_above(tri[2],quads);
  }

  return cgns::new_Elements(
    "PENTA_6",
    cgns::PENTA_6,
    std::move(prism_vtx),
    first_cell_id,first_cell_id+n_prism-1
  );
}

template<class Connecivity_type_0, class Connecivity_type_1> auto
share_vertices(const Connecivity_type_0& c0, const Connecivity_type_1& c1) -> bool {
  // TODO replace by std_e::set_intersection_size if size_0*size_1 >> size*log(size)
  for (auto x : c0) {
    for (auto y : c1) {
      if (x==y) return true;
    }
  }
  return false;
}

template<class I> auto
convert_to_hexa_vtx(const auto& hexa_face, const tree& ngons, I first_cell_id) -> tree {
  auto face_vtx = make_connectivity_range<I>(ngons);
  auto first_ngon_id = ElementRange<I>(ngons)[0];
  auto pe = ParentElements<I>(ngons);

  I n_hexa = hexa_face.size();
  I n_vtx = n_hexa*8;
  std::vector<I> hexa_vtx(n_vtx);
  auto d_first = hexa_vtx.data();

  for (int k=0; k<hexa_face.size(); ++k) {
    const auto& hexa = hexa_face[k];
    I hexa_id = first_cell_id+k;

    std::array<I,4> quad_0;
    I quad_0_idx = hexa[0]-first_ngon_id;
    auto quad_0_in_ngon = face_vtx[quad_0_idx];
    std::copy(begin(quad_0_in_ngon),end(quad_0_in_ngon),begin(quad_0));

    std::array<std_e::array<I,4>,4> side_quads;
    int side_quad_pos = 0;
    for (int i=0; i<6; ++i) {
      I quad_idx = hexa[i]-first_ngon_id;
      auto quad = face_vtx[quad_idx];
      STD_E_ASSERT(quad.size()==4);
    }
    for (int i=1; i<6; ++i) {
      I quad_idx = hexa[i]-first_ngon_id;
      auto quad = face_vtx[quad_idx];
      if (share_vertices(quad,quad_0)) {
        side_quads[side_quad_pos] = quad;
        ++side_quad_pos;
      }
    }

    // use quad_0 as nodes 1,2,3,4
    if (pe(quad_0_idx,0)==hexa_id) { // outward normal
      std::reverse(begin(quad_0),end(quad_0));
    } else {
      STD_E_ASSERT(pe(quad_0_idx,1)==hexa_id);
    }
    d_first = std::copy(begin(quad_0),end(quad_0),d_first);

    // nodes "above" quad_0
    *d_first++ = node_above(quad_0[0],side_quads);
    *d_first++ = node_above(quad_0[1],side_quads);
    *d_first++ = node_above(quad_0[2],side_quads);
    *d_first++ = node_above(quad_0[3],side_quads);
  }

  return cgns::new_Elements(
    "HEXA_8",
    cgns::HEXA_8,
    std::move(hexa_vtx),
    first_cell_id,first_cell_id+n_hexa-1
  );
}

template<class I> auto
convert_to_simple_volume_connectivities(const tree& ngons, const tree& nfaces, const std::vector<I> cell_partition_indices, I first_cell_id) -> std::vector<tree> {
  auto tet_face   = make_connectivity_subrange(nfaces,cell_partition_indices[0],cell_partition_indices[1]);
  auto pyra_face  = make_connectivity_subrange(nfaces,cell_partition_indices[1],cell_partition_indices[2]);
  auto prism_face = make_connectivity_subrange(nfaces,cell_partition_indices[2],cell_partition_indices[3]);
  auto hexa_face  = make_connectivity_subrange(nfaces,cell_partition_indices[3],cell_partition_indices[4]);

  std::vector<tree> cell_sections;

  I cur_cell_id = first_cell_id;
  if (tet_face.size()>0) {
    cell_sections.push_back(convert_to_tet_vtx(tet_face,ngons,cur_cell_id));
    cur_cell_id += tet_face.size();
  }

  if (pyra_face.size()>0) {
    cell_sections.push_back(convert_to_pyra_vtx(pyra_face,ngons,cur_cell_id));
    cur_cell_id += pyra_face.size();
  }

  if (prism_face.size()>0) {
    cell_sections.push_back(convert_to_prism_vtx(prism_face,ngons,cur_cell_id));
    cur_cell_id += prism_face.size();
  }

  if (hexa_face.size()>0) {
    cell_sections.push_back(convert_to_hexa_vtx(hexa_face,ngons,cur_cell_id));
    cur_cell_id += hexa_face.size();
  }

  return cell_sections;
}


template<class I> auto
_convert_zone_to_std_elements(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t");

  tree& ngons = element_section(z,cgns::NGON_n);
  tree& nfaces = element_section(z,cgns::NFACE_n);
  auto face_pls = cgns::get_zone_point_lists<I>(z,"FaceCenter");

  // partition faces and cells
  permute_boundary_ngons_at_beginning<I>(ngons,nfaces,face_pls);
  auto last_tri_index = partition_bnd_faces_by_number_of_vertices<I>(ngons,nfaces,face_pls);
  auto cell_partition_indices = partition_cells_into_simple_types<I>(ngons,nfaces);

  // convert to simple connectivities
  auto [bnd_elt_sections,next_avail_id] = convert_to_simple_exterior_boundary_connectivities(ngons,last_tri_index);
  auto vol_elt_sections = convert_to_simple_volume_connectivities(ngons,nfaces,cell_partition_indices,next_avail_id);

  // update tree
  rm_children_by_names(z,{name(ngons),name(nfaces)});

  emplace_children(z,std::move(bnd_elt_sections));
  emplace_children(z,std::move(vol_elt_sections));
}

auto
convert_zone_to_std_elements(tree& z) -> void {
  STD_E_ASSERT(label(z)=="Zone_t"); // TODO is_gc_maia_zone

  if (value(z).data_type()=="I4") return _convert_zone_to_std_elements<I4>(z);
  //if (value(z).data_type()=="I8") return _convert_zone_to_std_elements<I8>(z); // TODO
  throw cgns::cgns_exception("Zone "+name(z)+" has a value of data type "+value(z).data_type()+" but it should be I4 or I8");
}


} // maia
