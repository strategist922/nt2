cmake_minimum_required(VERSION 2.6)

set(Boost_ADDITIONAL_VERSIONS "1.46" "1.46.0" "1.46.1" "1.47" "1.47.0" "1.48" "1.48.0")
find_package(Boost 1.46.0 REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

add_executable(resource_compiler resource_compiler.cpp)
set_target_properties(resource_compiler
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
