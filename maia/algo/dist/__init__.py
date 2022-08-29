"""
Distributed algorithms for distributed trees
"""

from .conformize_jn              import conformize_jn_pair

from .duplicate                  import duplicate_from_periodic_jns,\
                                        duplicate_from_rotation_jns_to_360

from .elements_to_ngons          import elements_to_ngons
elements_to_poly = elements_to_ngons

from .extract_surf_dmesh         import extract_surf_tree_from_bc

from .merge                      import merge_connected_zones,\
                                        merge_zones

from .ngon_from_std_elements     import generate_ngon_from_std_elements

from .ngons_to_elements          import ngons_to_elements
poly_to_elements = ngons_to_elements

from .rearrange_element_sections import rearrange_element_sections

from .s_to_u                     import convert_s_to_u, convert_s_to_ngon
convert_s_to_poly = convert_s_to_ngon

from .vertex_list                import generate_jn_vertex_list,\
                                        generate_jns_vertex_list
