#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  std::cout << "Json read example\n";
  json settings;
  if (argc > 1) {
    std::cout << "Reading json from file " << argv[1] << "\n";
    std::ifstream input_file(argv[1]);
    input_file >> settings;
  } else {
    std::cout << "Reading json from standard input:\n";
    std::cin >> settings;
  }
  std::cout << "Simulation settings:\n\n";
  std::cout << settings.dump(4) << "\n\n";
  return 0;
}
