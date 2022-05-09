#pragma once

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
