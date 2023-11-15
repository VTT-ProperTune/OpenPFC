cmake_minimum_required(VERSION 3.15)
project(Aluminum VERSION 0.1.0 LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    find_package(OpenPFC REQUIRED)
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
    find_package(nlohmann_json REQUIRED)
    set(CMAKE_CXX_STANDARD 17)
    find_package(Catch2)
endif()

add_executable(aluminumNew aluminumNew.cpp)
target_link_libraries(aluminumNew OpenPFC nlohmann_json::nlohmann_json)

if(TARGET Catch2::Catch2WithMain)
    add_executable(aluminumTest aluminumTest.cpp)
    target_link_libraries(aluminumTest OpenPFC nlohmann_json::nlohmann_json Catch2::Catch2WithMain)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    catch_discover_tests(aluminumTest)
else()
    message(STATUS "Aluminum: Catch2 not found, not building tests")
endif()

install(TARGETS aluminumNew DESTINATION bin)
