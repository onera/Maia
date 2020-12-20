#include "std_e/unit_test/doctest.hpp"
#include "std_e/log.hpp"

#include <vector>

// Note Julien: la syntaxe "auto ma_fonction(...) -> type_de_retour" est équivalente à "type_de_retour ma_fonction(...)"
auto subset_sum_positions(int* first, int* last, int sum) -> std::pair<bool,std::vector<int*>> {
  if (sum==0) return {true,{}};

  std::vector<int*> candidates = {};
  int s = 0; 
  // NOTE Julien: pour les algorithmes bas-niveau, on veut généralement parcourir la sequence [first,last[.
  //              pour cela, on a besoin d'un itérateur "current" qui commence à "first" et qu'on incrémente
  //              on pourrait faire ça ici (ça serait peut-être plus clair)
  //              mais la convention est la suivante: on ne déclare pas "current", et on utilise "first" à la place
  //              (on peut le faire car first est passé par copie [i.e. le *pointeur* "first" est passé par valeur, pas la *case* où il pointe])
  while (first != last) { // loop until done

    // loop until the end and try to add elements to the candidates
    while (first != last) {
      if (s + *first == sum) { // we are done
        candidates.push_back(first);
        s += *first;
        return {true,candidates};
      }
      else if (s + *first < sum) { // add the position to the candidates and move forward
        candidates.push_back(first);
        s += *first;
        ++first;
      }
      else { // do not take this position in the candidates, just move forward
        ++first;
      }
    }

    // if we reach this point, no solution has been found yet
    // since nothing is found, it means one term in the sum is not right, so we need to pop it and restart from there
    if (candidates.size()>0) {
      first = candidates.back(); // restart from the last saved candidate...
      candidates.pop_back(); // ... remove it from the candidates...
      s -= *first; // ... and from the sum ...
      ++first; // ... and begin just after
    }
    // else: nothing to pop in the candidates, nothing matching, so do nothing: let the loop end
  }
  return {false,{}};

  // TODO: const-correctness
  // TODO: generalize to other types (templatize int* -> iterator)
  // TODO: extract into an algorithm that could be restarted in order to search other matching subsets
  // TODO: since the complexity is exponential, give max number of tries
  // TODO: generalize the matching function to allow for a tolerance
}


TEST_CASE("subset_sum") {
  std::vector<int> v = {3,9,8,4,5,7,10};
  auto [found,positions] = subset_sum_positions(v.data(),v.data()+v.size(),15);

  CHECK( found );

  CHECK( positions.size() == 3 );
  CHECK( *positions[0] == 3 );
  CHECK( *positions[1] == 8 );
  CHECK( *positions[2] == 4 );
}
