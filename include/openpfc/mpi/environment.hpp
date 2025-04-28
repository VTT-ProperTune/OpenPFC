// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_MPI_ENVIRONMENT_HPP
#define PFC_MPI_ENVIRONMENT_HPP

#include <mpi.h>

namespace pfc {
namespace mpi {

class environment {
public:
  inline environment();
  inline ~environment();
  inline std::string processor_name();
  inline bool initialized();
  inline bool finalized();
};

inline environment::environment() {
  MPI_Init(NULL, NULL);
}

inline environment::~environment() {
  MPI_Finalize();
}

inline std::string environment::processor_name() {
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlen;
  MPI_Get_processor_name(name, &resultlen);
  return std::string(name, resultlen);
}

inline bool environment::initialized() {
  int flag;
  MPI_Initialized(&flag);
  return (flag != 0);
}

inline bool environment::finalized() {
  int flag;
  MPI_Finalized(&flag);
  return (flag != 0);
}

} // namespace mpi
} // namespace pfc

#endif // PFC_MPI_ENVIRONMENT_HPP
