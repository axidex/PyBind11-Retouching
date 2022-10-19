include(FetchContent)

FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
)

message(STATUS "Checking dlib...")

FetchContent_MakeAvailable(dlib)