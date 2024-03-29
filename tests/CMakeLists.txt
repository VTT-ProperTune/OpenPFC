# https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#top

find_package(Catch2 REQUIRED)
add_executable(OpenPFCTests
               test_arraynd.cpp
               test_world.cpp
               test_decomposition.cpp
               test_discrete_field.cpp
               test_field_modifier.cpp
               test_fft.cpp
               test_ic_constant.cpp
               test_model.cpp
               test_multi_index.cpp
               test_simulator.cpp
               test_time.cpp
               )
target_link_libraries(OpenPFCTests PRIVATE OpenPFC Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(OpenPFCTests)
