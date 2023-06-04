# Minimum version
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14) # TODO: Change to 14?
endif()
# set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# Project settings
project(
  Mandelmagic
  VERSION 0.1.0
  DESCRIPTION ""
  LANGUAGES CXX C)

# Using g++ for now
set(CMAKE_CXX_COMPILER /usr/bin/g++)

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(example-app example-app.cpp) target_link_libraries(example-app
# "${TORCH_LIBRARIES}")

# look at testing
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# Testing
find_package(Catch2 3 REQUIRED)

# These tests can use the Catch2-provided main add_executable(tests
# test/lexer_test.cpp) target_link_libraries(tests PRIVATE
# Catch2::Catch2WithMain)

# set (CMAKE_CXX_FLAGS "-o2")

# target_compile_options(./build/flux PRIVATE -o3)

# name of executable would be flux
add_executable(mandel src/main.cpp)
