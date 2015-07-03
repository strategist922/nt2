################################################################################
#         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
#         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
#
#          Distributed under the Boost Software License, Version 1.0.
#                 See accompanying file LICENSE.txt or copy at
#                     http://www.boost.org/LICENSE_1_0.txt
################################################################################

find_package(OpenCL REQUIRED)

if(NOT OpenCL_FOUND)
  set(NT2_ARCH.OPENCL_DEPENDENCIES_FOUND 0)
endif()
set(NT2_ARCH.OPENCL_DEPENDENCIES_INCLUDE_DIR ${OpenCL_INCLUDE_DIRS})
set(NT2_ARCH.OPENCL_DEPENDENCIES_LIBRARIES ${OpenCL_LIBRARIES})
set(NT2_ARCH.OPENCL_COMPILE_FLAGS "-DNT2_HAS_OPENCL")
set(NT2_ARCH.OPENCL_DEPENDENCIES_EXTRA
#    boost.dispatch
#    core.base
#    core.sdk
   )
