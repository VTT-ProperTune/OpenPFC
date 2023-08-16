if(NOT Catch2_FIND_VERSION)
  set(Catch2_FIND_VERSION 3.3.2)
endif()

message(STATUS "Fetching Catch2 version ${Catch2_FIND_VERSION} from GitHub https://github.com/catchorg/Catch2.git")

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        "v${Catch2_FIND_VERSION}"
)

FetchContent_MakeAvailable(Catch2)
