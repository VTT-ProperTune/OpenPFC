#include <openpfc/world.hpp>

using namespace pfc;

int main() {
  World world({10, 20, 30}, {0.0, 0.0, 0.0}, {0.1, 0.1, 0.1});
  if (world.get_Lx() != 10) {
    return -1;
  }
  return 0;
}
