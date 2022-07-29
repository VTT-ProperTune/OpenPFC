# https://github.com/nlohmann/json#cmake
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)

add_executable(tungsten tungsten.cpp)
target_link_libraries(tungsten PRIVATE pfc nlohmann_json::nlohmann_json)
