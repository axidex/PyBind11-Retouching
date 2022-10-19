include(FetchContent)

FetchContent_Declare(
    wavelib
    GIT_REPOSITORY https://github.com/rafat/wavelib.git
)

message(STATUS "Checking wavelib...")

FetchContent_MakeAvailable(wavelib)