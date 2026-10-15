cmake_minimum_required(VERSION 3.25)
project(edgeproof_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(EDGEPROOF_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
    -Wold-style-cast -Wdouble-promotion)

# Geometry and decoding: no dependencies, unit-tested on their own.
add_library(edgeproof_yolo src/yolo.cpp)
target_include_directories(edgeproof_yolo PUBLIC include)
target_compile_options(edgeproof_yolo PRIVATE ${EDGEPROOF_WARNINGS})

enable_testing()
add_executable(test_yolo tests/test_yolo.cpp)
target_link_libraries(test_yolo PRIVATE edgeproof_yolo)
target_compile_options(test_yolo PRIVATE ${EDGEPROOF_WARNINGS})
add_test(NAME yolo COMMAND test_yolo)

# The runner: OpenCV for decoding and resizing images, ONNX Runtime for the network.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
include(OnnxRuntime)
add_executable(edgeproof-run src/main.cpp)
target_link_libraries(edgeproof-run PRIVATE edgeproof_yolo edgeproof::onnxruntime ${OpenCV_LIBS})
target_include_directories(edgeproof-run SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_options(edgeproof-run PRIVATE ${EDGEPROOF_WARNINGS})
