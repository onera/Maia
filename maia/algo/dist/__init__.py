"""
Distributed algorithms for distributed trees
"""

from .concat_nodes               import concatenate_subsets_from_families,\
                                        deconcatenate_subsets_from_families

from .conformize_jn              import conformize_jn_pair

from .connect_match              import connect_1to1_families

from .duplicate                  import duplicate_from_periodic_jns,\
                                        duplicate_from_rotation_jns_to_360,\
                                        duplicate_family_from_periodic_jns,\
                                        duplicate_family_from_rotation_jns_to_360

from .extrusion                  import extrude

from .merge                      import merge_all_zones_from_families,\
                                        merge_connected_zones,\
                                        merge_zones,\
                                        merge_zones_from_family

from .merge_degen_bc             import remove_degen_faces_from_family

from .mesh_adaptation            import adapt_mesh_with_feflo

from .mixed_to_std_elements      import convert_mixed_to_elements

from .multigrid                  import agglomerate_cells

from .ngon_from_std_elements     import convert_elements_to_ngon

from .ngons_to_elements          import convert_ngon_to_elements

from .redistribute               import redistribute_tree

from .retrieve_ridges            import find_ridges

from .s_to_u                     import convert_s_to_u, convert_s_to_ngon

from .sections_tools             import concatenate_elt_sections,\
                                        reorder_elt_sections_from_dim

from .std_elements_to_mixed      import convert_elements_to_mixed

from .vertex_list                import generate_jn_vertex_list,\
                                        generate_jns_vertex_list
