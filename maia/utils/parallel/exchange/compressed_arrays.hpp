#pragma once


#include <vector>


template<class T>
// requires T is bitwise copyable
struct compressed_arrays { // TODO RENAME cat_sequences
  std::vector<T> cat_arrays; // RENAME elements
  std::vector<int> sizes; // RENAME lengths
};


using serialized_strings = compressed_arrays<char>;
