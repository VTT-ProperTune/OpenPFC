add_executable(mpi_worker mpi_worker.cpp)
target_link_libraries(mpi_worker PRIVATE pfc)

add_executable(mpi_worker_inside_class mpi_worker_inside_class.cpp)
target_link_libraries(mpi_worker_inside_class PRIVATE pfc)

add_executable(mpi_timers mpi_timers.cpp)
target_link_libraries(mpi_timers PRIVATE pfc)

add_executable(world world.cpp)
target_link_libraries(world PRIVATE pfc)

add_executable(clean_tungsten clean_tungsten.cpp)
target_link_libraries(clean_tungsten PRIVATE pfc)

add_executable(decomposition decomposition.cpp)
target_link_libraries(decomposition PRIVATE pfc)

add_executable(decomposition_with_mpi decomposition_with_mpi.cpp)
target_link_libraries(decomposition_with_mpi PRIVATE pfc)

add_executable(distributed_fft distributed_fft.cpp)
target_link_libraries(distributed_fft PRIVATE pfc)

add_executable(diffusion_model diffusion_model.cpp)
target_link_libraries(diffusion_model PRIVATE pfc)

add_executable(clean_tungsten_multiple_seeds clean_tungsten_multiple_seeds.cpp)
target_link_libraries(clean_tungsten_multiple_seeds PRIVATE pfc)

add_executable(clean_tungsten_with_bc clean_tungsten_with_bc.cpp)
target_link_libraries(clean_tungsten_with_bc PRIVATE pfc)

add_executable(json_read json_read.cpp)
target_link_libraries(json_read PRIVATE pfc)
