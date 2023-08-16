add_executable(tungsten tungsten.cpp)
target_link_libraries(tungsten PRIVATE OpenPFC nlohmann_json::nlohmann_json)

option(TUNGSTEN_REUSE_ARRAYS "Reuse some arrays" OFF)
if(TUNGSTEN_REUSE_ARRAYS)
  message(STATUS "Tungsten: reusing arrays")
  target_compile_definitions(tungsten PUBLIC MAHTI_HACK)
endif()

add_subdirectory(aluminumNew)
