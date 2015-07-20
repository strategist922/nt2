################################################################################
#         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
#         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
#
#          Distributed under the Boost Software License, Version 1.0.
#                 See accompanying file LICENSE.txt or copy at
#                     http://www.boost.org/LICENSE_1_0.txt
################################################################################

include(NT2Module)
nt2_module_source_setup(arch.shared_memory)
nt2_module_postconfigure(gather_includes --name shared_memory.hpp
                                         nt2/sdk/
                                         --out nt2/sdk
                        )

set(script_path ${NT2_SOURCE_ROOT}/modules/arch/shared_memory/script/run_microbenchmarks.sh)
set(build_dir ${NT2_BINARY_DIR})

nt2_module_postconfigure(runtime_costs --path ${script_path} --build_dir ${build_dir})

add_custom_target(run_microbenchmarks
                  COMMAND ${script_path} ${build_dir}
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                  COMMENT "Update microbenchmarks"
                  )

add_library(foo STATIC foo.cpp)
