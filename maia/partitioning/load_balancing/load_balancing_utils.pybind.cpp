#include <vector>
#include "std_e/future/span.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

bool isSubsetSum(int set[], int n, int sum)
{
    // The value of subset[i][j] will be true if there is a subset of set[0..i-1] with sum equal to j
    bool subset[n + 1][sum + 1];
 
    // If sum is 0, then answer is true
    for (int i = 0; i <= n; i++)
        subset[i][0] = true;
 
    // If sum is not 0 and set is empty,
    // then answer is false
    for (int i = 1; i <= sum; i++)
        subset[0][i] = false;
 
    // Fill the subset table in botton up manner
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= sum; j++) {
            if (j < set[i - 1])
                subset[i][j] = subset[i - 1][j];
            if (j >= set[i - 1])
                subset[i][j] = subset[i - 1][j] || subset[i - 1][j - set[i - 1]];
        }
    }
    return subset[n][sum];
}

void subset_sum_recursive(std_e::span<const int>numbers,
                          int target,
                          const std::vector<const int*>& partial, 
                          std::vector<const int*>& solutions,
                          bool &found)
{
  //Sum array using lambda func -> sum content of pointers
  int s = accumulate(partial.begin(), partial.end(), 0, [](int acc, const int* x) {return acc + *x;});

  if (found)
    return;
  if(s == target) {
    solutions = partial;
    found = true;
  }
  if(s >= target)
    return;

  auto first = numbers.data();
  auto last  = numbers.data() + numbers.size();
  for (auto ai = first; ai != last; ++ai) {
    auto remaining = std_e::make_span(ai+1, last);
    auto partial_rec = partial;
    partial_rec.push_back(ai);
    if (found)
      break;
    else
      subset_sum_recursive(remaining, target, partial_rec, solutions, found);
  }
}

std::vector<int> subset_sum(const std::vector<int>& numbers, int target) {

  std::vector<const int*> solution;
  bool found = false;
  subset_sum_recursive(std_e::make_span(numbers), target, {}, solution, found);

  std::vector<int> indices;
  for (auto x: solution)
    indices.push_back(x - numbers.data()); //Get indices

  return indices;
}

auto search_match(py::array_t<int> ordered_list, int target, int max_tries) -> py::list {

  assert(ordered_list.ndim() == 1);
  int N = ordered_list.size();
  if (N==0) {
    return py::cast(std::vector<int>{});
  }

  const int *np_data = ordered_list.data(0);

  if (!std::is_sorted(np_data, np_data+N)) {
    return py::cast(std::vector<int>{}); // We shoud raise here, list is not sorted
  }

  auto upper = std::upper_bound(np_data, np_data+N, target);

  std::vector<int> input;
  std::copy(np_data, upper, std::back_inserter(input)); //Copy data up to max bound

  std::vector<int> indices = subset_sum(input, target);

  return py::cast(indices); //Return list
}

PYBIND11_MODULE(load_balancing_utils, m) {
  m.doc() = "C++ maia functions wrapped by pybind";

  m.def("search_match", &search_match, "search_match");
}

//Old version, returns all combinations
//bool subset_sum_recursiveOLD(const std::vector<int>& numbers, int target,
                          //const std::vector<int>& partial, std::vector<int>& solutions) {
  //int s = 0;
  //for (std::vector<int>::const_iterator cit = partial.begin(); cit != partial.end(); cit++) {
    //s += *cit;
  //}
  //std::cout << "entering func with s = " << s << std::endl;
  //if(s == target) {
    //std::cout << "solution found " << std::endl;
    //solutions.push_back(partial.size());
    //for (std::vector<int>::const_iterator cit = partial.begin(); cit != partial.end(); cit++) {
      //solutions.push_back(*cit);
    //}
    //return true;
  //}
  //if(s >= target)
    //return false;

  //int n;
  //for (std::vector<int>::const_iterator ai = numbers.begin(); ai != numbers.end(); ai++) {
    //n = *ai;
    //std::vector<int> remaining;
    //for(std::vector<int>::const_iterator aj = ai; aj != numbers.end(); aj++) {
      //if(aj == ai)continue;
      //remaining.push_back(*aj);
    //}
    //std::vector<int> partial_rec=partial;
    //partial_rec.push_back(n);
    //bool found = subset_sum_recursiveOLD(remaining,target,partial_rec, solutions);
    //std::cout << "found is " << std::boolalpha << found <<std::endl;
    //if (found)
      //break;
  //}
  //return true;
//}

