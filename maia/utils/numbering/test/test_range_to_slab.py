from maia.utils.numbering import range_to_slab

def test_cell_to_indexes():
  assert range_to_slab.cell_to_indexes(1, 80, 8) == (1,0,0)
  assert range_to_slab.cell_to_indexes(8, 80, 8) == (0,1,0)
  assert range_to_slab.cell_to_indexes(81, 80, 8) == (1,0,1)
  assert range_to_slab.cell_to_indexes(273, 80, 8) == \
      (273 - 3*80 - 4*8, (273-3*80)//8, 273//80)

class Test_compute_slabs():
  def test_compute_slabs(self):
    slabs = range_to_slab.compute_slabs([5,5,5], [19, 62])
    assert len(slabs) == 5
    assert slabs[0] == [[4, 5], [3, 4], [0, 1]]
    assert slabs[1] == [[0, 5], [4, 5], [0, 1]]
    assert slabs[2] == [[0, 5], [0, 5], [1, 2]]
    assert slabs[3] == [[0, 5], [0, 2], [2, 3]]
    assert slabs[4] == [[0, 2], [2, 3], [2, 3]]

  def test_compute_slabs_combine(self):
    slabs = range_to_slab.compute_slabs([7,3,9], [105, 161])
    assert len(slabs) == 2
    assert slabs[0] == [[0, 7], [0, 3], [5, 7]]
    assert slabs[1] == [[0, 7], [0, 2], [7, 8]]
