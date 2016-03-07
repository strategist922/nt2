################################################################################
#         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
#
#          Distributed under the Boost Software License, Version 1.0.
#                 See accompanying file LICENSE.txt or copy at
#                     http://www.boost.org/LICENSE_1_0.txt
################################################################################
cmake_minimum_required(VERSION 2.6)

add_executable(kernel_gen kernel_gen.cpp)


set_target_properties(kernel_gen
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
                                 RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_BINARY_DIR}
                                 COMPILE_FLAGS   "-std=c++11"
                     )
