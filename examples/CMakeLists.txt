add_executable(mpi_worker mpi_worker.cpp)
target_link_libraries(mpi_worker PRIVATE pfc)

add_executable(clean_tungsten clean_tungsten.cpp)
target_link_libraries(clean_tungsten PRIVATE pfc)
