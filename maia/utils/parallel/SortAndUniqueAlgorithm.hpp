#ifndef _GLOBGRAPH_UTILITIES_SORTANDUNIQUEALGORITHM_H_
#define _GLOBGRAPH_UTILITIES_SORTANDUNIQUEALGORITHM_H_

// ------------------------------------------------------------------
// External include
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// Internal include

// ------------------------------------------------------------------

// ===========================================================================
template<class ForwardIt, class T, class Compare = std::less<> >
  ForwardIt binary_find(ForwardIt first, ForwardIt last, const T& value, Compare comp={})
{
  // Note: BOTH type T and the type after ForwardIt is dereferenced
  // must be implicitly convertible to BOTH Type1 and Type2, used in Compare.
  // This is stricter than lower_bound requirement (see above)

  //first = std::lower_bound(first, last, value, comp);
  //return first != last && !comp(value, *first) ? first : last;
  return std::lower_bound(first, last, value, comp);
}

#endif /* _GLOBGRAPH_UTILITIES_SORTANDUNIQUEALGORITHM_H_ */
