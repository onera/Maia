#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

bool isSubsetSum(const int set[], const int n, const int sum)
{
  /*
   * Checks if a solution to the subset sum problem exists
  */
  // The value of subset[i][j] will be true if there is a subset of set[0..i-1] with sum equal to j
  int sizeX = n+1;
  int sizeY = sum+1;
  int max_alloc = 100000000; // Exit if arrays is bigger than that
  if (sizeY > (max_alloc / sizeX)) {
    return false;
  }
  bool* subset = new bool[sizeX*sizeY]; // subset[n + 1][sum + 1];
  // Access using Subset[i][j] = subset[i*sizeY+j]

  // If sum is 0, then answer is true
  for (int i = 0; i <= n; i++)
      subset[i*sizeY] = true; // subset[i][0]

  // If sum is not 0 and set is empty,
  // then answer is false
  for (int i = 1; i <= sum; i++)
      subset[i] = false; //subset[0][i]

  // Fill the subset table in botton up manner
  for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= sum; j++) {
          if (j < set[i - 1])
              subset[i*sizeY+j] = subset[(i - 1)*sizeY + j]; //subset[i][j] = subset[i-1][j]
          if (j >= set[i - 1])
              // subset[i][j] = subset[i - 1][j] || subset[i - 1][j - set[i - 1]];
              subset[i*sizeY+j] = subset[(i - 1)*sizeY+j] || subset[(i - 1)*sizeY + (j - set[i - 1])];
      }
  }
  bool out = subset[n*sizeY + sum]; // subset[n][sum]
  delete []subset;
  return out;
}

auto subset_sum_positions(int* first, int* last, int target, int max_it) -> std::pair<bool,std::vector<int*>> {
  /*
   * Return a solution to the subset sum problem.
  */
  // TODO: const-correctness
  // TODO: generalize to other types (templatize int* -> iterator)
  // TODO: extract into an algorithm that could be restarted in order to search other matching subsets
  // TODO: generalize the matching function to allow for a tolerance

  if (target==0) return {true,{}};
  if (std::accumulate(first, last, 0) < target) return {false, {}};

  std::vector<int*> candidates = {};
  int s = 0;
  int it = 0;
  while (first != last && it < max_it) { // loop until done

    // loop until the end and try to add elements to the candidates
    while (first != last) {
      ++it;
      if (s + *first == target) { // we are done
        candidates.push_back(first);
        s += *first;
        return {true,candidates};
      }
      else if (s + *first < target) { // add the position to the candidates and move forward
        candidates.push_back(first);
        s += *first;
        ++first;
      }
      else { // do not take this position in the candidates, just move forward
        ++first;
      }
    }

    // if we reach this point, no solution has been found yet
    // since nothing is found, it means one term in the target is not right, so we need to pop it and restart from there
    if (candidates.size()>0) {
      first = candidates.back(); // restart from the last saved candidate...
      candidates.pop_back(); // ... remove it from the candidates...
      s -= *first; // ... and from the target ...
      ++first; // ... and begin just after
    }
    // else: nothing to pop in the candidates, nothing matching, so do nothing: let the loop end
  }
  return {false,{}};
}

auto search_subset_match(py::array_t<int> sorted_np, int target, int max_it) -> py::list {
  /*
   * Bind the subset sum problem for a numpy int array : return (if existing)
   * a list of indices such that array[indices].sum == target.
   * The input array must be sorted. A maximal number of iterations can be specified.
  */

  if (sorted_np.ndim() != 1 || sorted_np.size()==0)
    return py::cast(std::vector<int>{});

  int N = sorted_np.size();
  const int *np_data = sorted_np.data(0);

  if (!std::is_sorted(np_data, np_data+N)) {
    return py::cast(std::vector<int>{}); // We shoud raise here, list is not sorted
  }

  // First index >= target
  auto upper = std::upper_bound(np_data, np_data+N, target);

  //Array is sorted, we can juste search beetween begin and upper
  std::vector<int> indices;
  if (isSubsetSum(np_data, upper - np_data, target)) {
    auto result = subset_sum_positions((int *) np_data, (int *) upper, target, max_it);
    for (auto x: result.second)
      indices.push_back(x - np_data); //Get indices
  }
  return py::cast(indices); //Return list
}
