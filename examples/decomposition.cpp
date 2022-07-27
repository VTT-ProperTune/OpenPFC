#include <iostream>

#include <pfc/decomposition.hpp>

using namespace std;
using namespace pfc;

int main() {
  Decomposition d1({32, 4, 4}, 0, 2);
  cout << d1 << endl;
  return 0;
}
