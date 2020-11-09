from maia.utils import parse_yaml_cgns
import Converter.Internal as I
import numpy as np

with open("/scratchm/bberthou/travail/git_all_projects/example.yaml") as yt:
  t = parse_yaml_cgns.to_pytree(yt)

  I.printTree(t)

  #assert t == [['Base0', [3, 3], [['Zone0', [24, 6, 0], [['GridCoordinates', None, [['CoordinateX', np.array([0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.]), [], 'DataArray_t'], ['CoordinateY', np.array([0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2., 0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.], dtype=np.float32), [], 'DataArray_t'], ['CoordinateZ', np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=np.float32), [], 'DataArray_t']], 'GridCoordinates_t'], ['ZoneBC', None, [['Inlet', None, [['GridLocation', 'FaceCenter', [], 'GridLocation_t'], ['PointList', [1, 2], [], 'IndexArray_t']], 'BC_t']], 'ZoneBC_t'], ['ZoneGridConnectivity', None, [['MixingPlane', 'Zone1', [['GridConnectivityType', 'Abutting1to1', [], 'GridConnectivityType_t'], ['GridLocation', 'FaceCenter', [], 'GridLocation_t'], ['PointList', [7], [], 'IndexArray_t'], ['PointListDonor', [1], [], 'IndexArray_t']], 'GridConnectivity']], 'ZoneGridConnectivity_t']], 'Zone_t']], 'Base_t']]


"""
['Base0',[3, 3],[1 son],'Base_t']
   |_['Zone0',[24, 6, 0],[3 sons],'Zone_t']
       |_['GridCoordinates',None,[3 sons],'GridCoordinates_t']
       |   |_['CoordinateX',array(shape=(24,),dtype='float64',order='F'),[0 son],'DataArray_t']
       |   |_['CoordinateY',array(shape=(24,),dtype='float32',order='F'),[0 son],'DataArray_t']
       |   |_['CoordinateZ',array(shape=(24,),dtype='float32',order='F'),[0 son],'DataArray_t']
       |_['ZoneBC',None,[1 son],'ZoneBC_t']
       |   |_['Inlet',None,[2 sons],'BC_t']
       |       |_['GridLocation','FaceCenter',[0 son],'GridLocation_t']
       |       |_['PointList',[1, 2],[0 son],'IndexArray_t']
       |_['ZoneGridConnectivity',None,[1 son],'ZoneGridConnectivity_t']
           |_['MixingPlane','Zone1',[4 sons],'GridConnectivity']
               |_['GridConnectivityType','Abutting1to1',[0 son],'GridConnectivityType_t']
               |_['GridLocation','FaceCenter',[0 son],'GridLocation_t']
               |_['PointList',[7],[0 son],'IndexArray_t']
               |_['PointListDonor',[1],[0 son],'IndexArray_t']
"""
