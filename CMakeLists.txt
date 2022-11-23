cmake_minimum_required(VERSION 3.16)
project("pybind-smoothing")
#set(ENGINE_PROJECT_NAME smoothingmodule)

if (UNIX)
    set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall --std=c++20 -fPIC" )
    set( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -fPIC" )
endif (UNIX)

set(CMAKE_CXX_STANDARD 20)

set(MODULE_SOURCE "${PROJECT_SOURCE_DIR}/src/module")


# pybind11
add_subdirectory(external/pybind11)
add_subdirectory(external/dlib)
add_subdirectory(external/tinyspline)
add_subdirectory(external/wavelib)


set(FILTER_PUBLIC_INC
    src/module/newfilter.h
)
set(FILTER_PUBLIC_SRC
    src/module/newfilter.cpp
)

set(CONVERTER_PUBLIC_SRC
    src/module/ndarray_converter.cpp
)

add_library(Filter STATIC ${FILTER_PUBLIC_SRC})
target_include_directories(Filter PRIVATE src/module/)
target_link_libraries(Filter PRIVATE ${OpenCV_LIBS})
set_target_properties(Filter PROPERTIES LINKER_LANGUAGE CXX)

include_directories(${PYTHON_INCLUDE_DIRS} ${_numpy_include_dirs})
message("Python libs: ${PYTHON_LIBRARIES}")

#module

pybind11_add_module (
    smoothingmodule
    ${MODULE_SOURCE}/module.cpp
)

target_include_directories (
    smoothingmodule
    PRIVATE ${MODULE_SOURCE}
)

FIND_PACKAGE(Python3 COMPONENTS Interpreter Development NumPy)

# opencv

find_package(OpenCV REQUIRED)
set(OpenCV_STATIC ON)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(Converter STATIC ${CONVERTER_PUBLIC_SRC})
target_link_libraries(Converter PRIVATE ${OpenCV_LIBS} pybind11::embed Python3::Python -static Python3::NumPy)
set_target_properties(Converter PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(smoothingmodule PRIVATE pybind11::embed Python3::Python -static Python3::NumPy dlib::dlib ${OpenCV_LIBS} tinysplinecxx wavelib Filter Converter)
target_include_directories(smoothingmodule PRIVATE external/wavelib/header)
target_compile_features(smoothingmodule PRIVATE $<IF:$<PLATFORM_ID:Windows>,cxx_std_23,cxx_std_20>) # to enable `/std:c++latest` 
