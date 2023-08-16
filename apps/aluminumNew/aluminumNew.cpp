#include "Aluminum.hpp"
#include "SeedGridFCC.hpp"

int main(int argc, char *argv[]) {
  std::cout << std::fixed;
  std::cout.precision(3);
  register_field_modifier<SeedGridFCC>("seed_grid_fcc");
  App<Aluminum> app(argc, argv);
  return app.main();
}
