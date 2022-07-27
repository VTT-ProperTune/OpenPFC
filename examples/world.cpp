#include <iostream>
#include <pfc/world.hpp>

using namespace pfc;
using namespace std;

int main() {
  World w({128, 128, 128}, {-64.0, -64.0, -64.0}, {0.5, 0.5, 0.5});
  cout << w << endl;
  return 0;
}
