##############################################################################
###   Copyright 2003 and onward LASMEA UMR 6602 CNRS/U.B.P Clermont-Ferrand
###   Copyright 2009 and onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
###
###          Distributed under the Boost Software License, Version 1.0
###                 See accompanying file LICENSE.txt or copy at
###                     http://www.boost.org/LICENSE_1_0.txt
##############################################################################

find_package(CUDA)

IF(CUDA_CUFFT_LIBRARIES)

include_directories(${CUDA_INCLUDE_DIRS})
link_libraries(${CUDA_CUFFT_LIBRARIES})

  set ( SOURCES
        fft.cpp
      )

add_definitions("-DNT2_USE_CUFFTW")

endif()

nt2_module_add_tests(arch.cuda.signal.unit ${SOURCES})

