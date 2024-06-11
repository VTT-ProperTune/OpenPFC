/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

/*
This is just a proposal on how to run application: either

```bash
./app input.json
```

or

```bash
cat input.json | ./app
```

or e.g. from slurm batch script:

```bash
#!/bin/bash

#SBATCH --partition=...
#SBATCH --job-name=...
#SBATCH --ntasks-per-node=...
#SBATCH --cpus-per-task=1
#SBATCH --nodes=...
#SBATCH --mem-per-cpu=...
#SBATCH --time=...
#SBATCH --output=%x-%j.log

srun ./app <<EOF
{
    "Lx": 1024,
    "Ly": 512,
    "Lz": 512,
    "x0": 0.0,
    "y0": 0.0,
    "z0": 0.0,
    "dx": 1.1107207345395915,
    "dy": 1.1107207345395915,
    "dz": 1.1107207345395915,
    "t0": 0.0,
    "t1": 10000000.0,
    "dt": 1.0,
    "initial_condition": "regular_grid",
    "results_dir": "data",
    "saveat": 1000.0
}
EOF
```

*/

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
