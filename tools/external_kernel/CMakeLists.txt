cmake_minimum_required(VERSION 2.6)

find_package(Boost 1.57.0 COMPONENTS system)

if(NOT ${Boost_FOUND})
message("-- [nt2.tool.external_kernel]  warning : dependencies not met, skipping")
else()

set(NT2_DIR ${CMAKE_SOURCE_DIR}/../../cmake)
find_package(NT2)
#list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(NOT WIN32)
link_libraries(dl)
endif()

include_directories(${CMAKE_SOURCE_DIR}/../gather_includes ${Boost_INCLUDE_DIRS}  .)
link_libraries(${Boost_LIBRARIES})

#TODO: is this the problem?
message(STATUS "----${NT2_FOUND_COMPONENTS}")
add_executable(external_kernel_gen external_kernel.cpp parse_symbol.cpp backend.cpp)

foreach(component ${NT2_FOUND_COMPONENTS})
string(REPLACE . / module ${component} )
file(GLOB backends RELATIVE ${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel ${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel/backends/*/)
message(STATUS "${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel ${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel/backends/*")

message(STATUS "backends=${backends}")
foreach(backend ${backends})
message(STATUS "backend :" ${backends})

if(IS_DIRECTORY ${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel/${backend})
message(STATUS "add extra directory ${backend}")
add_subdirectory(${NT2_SOURCE_ROOT}/modules/${module}/tools/external_kernel/${backend} ${CMAKE_BINARY_DIR}/${backend})
endif()
endforeach()
endforeach()

endif()

