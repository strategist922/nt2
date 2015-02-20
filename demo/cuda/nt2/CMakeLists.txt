################################################################################
#         Copyright 2014 - 2015 LRI    UMR 8623 CNRS/Univ Paris Sud XI
#
#          Distributed under the Boost Software License, Version 1.0.
#                 See accompanying file LICENSE.txt or copy at
#                     http://www.boost.org/LICENSE_1_0.txt
################################################################################

include_directories(${BOOST_ROOT})

add_executable(cuda_bandwidth_bench_nt2 cuda_bandwidth_bench_nt2.cpp)
add_executable(cuda_nt2 cuda_nt2.cpp)
add_executable(cuda_trsm cuda_trsm.cpp)

