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

#ifndef PFC_MPI_ENVIRONMENT_HPP
#define PFC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace pfc {
namespace mpi {

class environment {
public:
  environment();
  ~environment();
  std::string processor_name();
  bool initialized();
  bool finalized();
};

environment::environment() {
  MPI_Init(NULL, NULL);
}

environment::~environment() {
  MPI_Finalize();
}

std::string environment::processor_name() {
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name(name, &resultlen);
  return std::string(name, resultlen);
}

bool environment::initialized() {
  int flag;
  MPI_Initialized(&flag);
  return (flag != 0);
}

bool environment::finalized() {
  int flag;
  MPI_Finalized(&flag);
  return (flag != 0);
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_ENVIRONMENT_HPP
