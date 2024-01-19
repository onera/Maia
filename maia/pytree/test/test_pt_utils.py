
import pytest
import numpy as np

import maia.pytree as PT

from maia.pytree import utils

def test_gc_transform():
  t_matrix = np.array([[0,-1,0], [-1,0,0], [0,0,-1]])
  start_1 = np.array([17,3,1])
  start_2 = np.array([7,9,5])
  assert (utils._gc_transform_point(start_1, start_1, start_2, t_matrix) == start_2).all() #Start
  assert (utils._gc_transform_point(np.array([17,6,3]), start_1, start_2, t_matrix) == [4,9,3]).all() #Middle
  assert (utils._gc_transform_point(np.array([17,9,5]), start_1, start_2, t_matrix) == [1,9,1]).all() #End

  gc = PT.new_GridConnectivity1to1(point_range=[[17,17], [3,9], [1,5]], 
                                   point_range_donor=[[7,1],[9,9],[5,1]], 
                                   transform=[-2,-1,-3])

  assert (utils.gc_transform_point(gc, np.array([17,6,3])) == [4,9,3]).all()
  assert (utils.gc_transform_window(gc, np.array([[17,17], [4,8], [1,5]])) == [[6,2],[9,9],[5,1]]).all()