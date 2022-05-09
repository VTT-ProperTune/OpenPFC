include(FetchContent)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY "https://github.com/p-ranav/argparse.git"
  GIT_TAG "v2.2")

FetchContent_MakeAvailable(argparse)
