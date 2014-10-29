################################################################################
#         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
#         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
#
#          Distributed under the Boost Software License, Version 1.0.
#                 See accompanying file LICENSE.txt or copy at
#                     http://www.boost.org/LICENSE_1_0.txt
################################################################################

include(nt2.boost)

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # C++11 required
  if(MINGW)
    # somehow without that MinGW stops declaring Microsoft-specific CRT functions
    set(NT2_BOOST.DISPATCH_DEPENDENCIES_COMPILE_FLAGS "-std=gnu++0x")
  else()
    set(NT2_BOOST.DISPATCH_DEPENDENCIES_COMPILE_FLAGS "-std=c++0x")
  endif()
endif()

SET(NT2_BOOST.DISPATCH_DEPENDENCIES_FOUND ${Boost_FOUND})
SET(NT2_BOOST.DISPATCH_DEPENDENCIES_INCLUDE_DIR ${Boost_INCLUDE_DIRS})
SET(NT2_BOOST.DISPATCH_DEPENDENCIES_LIBRARY_DIR ${Boost_LIBRARY_DIRS})
