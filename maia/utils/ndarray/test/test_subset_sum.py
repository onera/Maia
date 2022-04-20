from cmaia.utils import search_subset_match

def test_search_match():
  assert search_subset_match([10,20,50,70], 60, 2**20) == [0,2]
  assert search_subset_match([10,20,50,70], 70, 2**20) == [1,2]
  assert search_subset_match([10,20,50,70], 32, 2**20) == []
  #assert load_balancing_utils.search_match([10,20,50,70], 32, tol=2)  == [0,1]
