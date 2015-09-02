##############################################################################
###   Copyright 2003 and onward LASMEA UMR 6602 CNRS/U.B.P Clermont-Ferrand
###   Copyright 2009 and onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
###
###          Distributed under the Boost Software License, Version 1.0
###                 See accompanying file LICENSE.txt or copy at
###                     http://www.boost.org/LICENSE_1_0.txt
##############################################################################

#add_library(generatedopencl OBJECT generated.cpp)

SET( SOURCES
    opencl_buffer.cpp
    specific_opencl.cpp
    transform.cpp
    blacknscholes.cpp
#    generated_cl.cpp
#    generated.cpp
   )

nt2_module_add_tests(arch.opencl.general.unit ${SOURCES})
add_library(generatedcl OBJECT generated.cpp)
#target_link_libraries(arch.opencl.general.unit.transform generatedopencl)
