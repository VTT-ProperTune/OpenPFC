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

#include <openpfc/results_writer.hpp>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace pfc;

int main(int argc, char *argv[]) {
  cout << fixed;
  cout.precision(3);
  MPI_Init(&argc, &argv);
  int me, num_ranks;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &num_ranks);
  if (num_ranks < 2) {
    if (me == 0) {
      cerr << "Run at least with two mpi processes\n";
    }
    MPI_Abort(comm, -1);
    return -1;
  }

  if (me == 0) cout << "Writing results with two separate mpi processes" << endl;
  {
    ResultsWriter *writer = new BinaryWriter("test_%04d.bin");
    if (me == 0) {
      writer->set_domain({8, 1, 1}, {4, 1, 1}, {0, 0, 0});
      writer->write(5, vector<double>{1, 2, 3, 4});
    } else if (me == 1) {
      writer->set_domain({8, 1, 1}, {4, 1, 1}, {4, 0, 0});
      writer->write(5, vector<double>{5, 6, 7, 8});
    } else {
      cout << "MPI rank " << me << " is idling" << endl;
    }
  }

  if (me == 0) {
    // let's read the data back
    fstream file("test_0005.bin", ios::in | ios::binary);
    vector<double> data(8);
    file.read((char *)(data.data()), sizeof(double) * 8);
    file.close();
    for (int i = 0; i < 8; i++) {
      cout << "data[" << i << "] = " << data[i] << endl;
    }
  }

  MPI_Finalize();

  return 0;
}
