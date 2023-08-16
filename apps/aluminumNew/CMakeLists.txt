cmake_minimum_required(VERSION 3.15)
project(Aluminum)
find_package(OpenPFC REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

find_package(nlohmann_json REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(aluminumNew aluminumNew.cpp)
target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json)
