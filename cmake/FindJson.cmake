include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY "https://github.com/nlohmann/json.git"
  GIT_TAG "v3.10.4")

FetchContent_MakeAvailable(json)
