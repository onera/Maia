import pytest
import numpy as np
import os
#mport Converter 

know_cassiopee = True
try:
  import Converter
except ImportError:
  know_cassiopee = False

import maia.pytree as PT

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_zone_tree():
  from maia.io import _hdf_io_cass as LC
  yt = """
Zone Zone_t:
  Hexa Elements_t [17, 0]:
    ElementConnectivity DataArray_t None:
  ZBC ZoneBC_t:
    bc BC_t "farfield":
      PointList IndexArray_t None:
    bc_withds BC_t "farfield":
      PointList IndexArray_t None:
      BCDataSet BCDataSet_t:
        PointList IndexArray_t None:
  ZGC ZoneGridConnectivity_t:
    gc GridConnectivity_t:
      PointList IndexArray_t None:
      PointListDonor IndexArray_t None:
  ZSR ZoneSubRegion_t:
    PointList IndexArray_t None:
  FS FlowSolution_t:
  FSPL FlowSolution_t:
    PointList IndexArray_t None:
"""
  zone = PT.yaml.to_node(yt)
  size_data = {'/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
               '/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
               '/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
               '/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
               '/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
               '/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
               '/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
               '/Zone/FSPL/PointList' : (1, 'I4', (1,10)),
              }

  LC.add_sizes_to_zone_tree(zone, '/Zone', size_data)

  assert PT.get_node_from_path(zone, 'Hexa/ElementConnectivity#Size') is None

  assert (PT.get_node_from_path(zone, 'ZBC/bc/PointList#Size')[1] == [1,30]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/PointList#Size')[1] == [1,100]).all()
  assert (PT.get_node_from_path(zone, 'ZBC/bc_withds/BCDataSet/PointList#Size')[1] == [1,10]).all()

  assert (PT.get_node_from_path(zone, 'ZGC/gc/PointList#Size')[1] == [1,20]).all()

  assert (PT.get_node_from_path(zone, 'ZSR/PointList#Size')[1] == [1,34]).all()

  assert (PT.get_node_from_path(zone, 'FSPL/PointList#Size')[1] == [1,10]).all()
  assert (PT.get_node_from_path(zone, 'FS/PointList#Size') is None)

@pytest.mark.skipif(not know_cassiopee, reason="Require Cassiopee")
def test_add_sizes_to_tree():
  from maia.io import _hdf_io_cass as LC
  yt = """
BaseA CGNSBase_t:
  Zone Zone_t:
    Hexa Elements_t [17, 0]:
      ElementConnectivity DataArray_t None:
    ZBC ZoneBC_t:
      bc BC_t "farfield":
        PointList IndexArray_t None:
      bc_withds BC_t "farfield":
        PointList IndexArray_t None:
        BCDataSet BCDataSet_t:
          PointList IndexArray_t None:
    ZGC ZoneGridConnectivity_t:
      gc GridConnectivity_t:
        PointList IndexArray_t None:
        PointListDonor IndexArray_t None:
    ZSR ZoneSubRegion_t:
      PointList IndexArray_t None:
"""
  tree = PT.yaml.to_cgns_tree(yt)
  size_data_tree = {'/BaseA/Zone/Hexa/ElementConnectivity' : (1, 'I4', 160),
                    '/BaseA/Zone/ZBC/bc/PointList' : (1, 'I4', (1,30)),
                    '/BaseA/Zone/ZBC/bc_withds/PointList' : (1, 'I4', (1,100)),
                    '/BaseA/Zone/ZBC/bc_withds/BCDataSet/PointList' : (1, 'I4', (1,10)),
                    '/BaseA/Zone/ZGC/gc/PointList' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZGC/gc/PointListDonor' : (1, 'I4', (1,20)),
                    '/BaseA/Zone/ZSR/PointList' : (1, 'I4', (1,34)),
                   }
  LC.add_sizes_to_tree(tree, size_data_tree)
  assert len(PT.get_nodes_from_name(tree, '*#Size')) == 6




# # Fixture pour simuler un fichier HDF de test
# @pytest.fixture
# def sample_hdf_file(tmpdir):
#     # Crée un arbre CGNS simple
#     root = PT.new_node("ParentNode", "UserDefinedData_t", 3.14)
#     zone = PT.new_node('Zone1','Zone_t', value=np.array([1, 2, 3]))
#     base = PT.new_node('Base1')
#     PT.add_child(base, zone)
#     PT.add_child(root,base)
    
#     # Sauvegarde l'arbre dans un fichier HDF temporaire
#     filename = os.path.join(tmpdir, 'test.hdf')
#     Converter.PyTree.convertPytree2File (tree, filename, format='bin_hdf')
#     return filename
  
# #pytest_parallel.mark.parallel(3)
# def test_load_size_tree(sample_hdf_file):
#     from maia.io import _hdf_io_cass as LC
#     comm= MPI.COMM_WORLD  # Communicateur MPI

#     # Appel de la fonction à tester
#     size_tree = load_size_tree(sample_hdf_file, comm)

#     # Vérifications
#     if comm.Get_rank() == 0:
#         # Sur le rang 0, l'arbre doit être chargé et traité
#         assert size_tree is not None
#         assert PT.get_node_from_name(size_tree, 'Base1') is not None
#         assert PT.get_node_from_name(size_tree, 'Zone1') is not None
#     else:
#         # Sur les autres rangs, l'arbre doit être diffusé depuis le rang 0
#         assert size_tree is not None
#         assert PT.get_node_from_name(size_tree, 'Base1') is not None
#         assert PT.get_node_from_name(size_tree, 'Zone1') is not None