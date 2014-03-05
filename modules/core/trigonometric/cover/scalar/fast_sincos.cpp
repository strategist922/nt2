//==============================================================================
//         Copyright 2003 - 2014   LASMEA UMR 6602 CNRS/UBP
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
// cover for functor fast_sincos in scalar mode
#include <nt2/trigonometric/include/functions/fast_sincos.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/simd/sdk/simd/io.hpp>
#include <cmath>
#include <iostream>
#include <nt2/include/constants/pio_4.hpp>
#include <nt2/include/functions/scalar/sincos.hpp>
#include <nt2/sdk/unit/args.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/cover.hpp>
#include <vector>

extern "C" {extern long double cephes_cosl(long double);}
extern "C" {extern long double cephes_sinl(long double);}

NT2_TEST_CASE_TPL(fast_sincos_0,  NT2_SIMD_REAL_TYPES)
{
  using nt2::unit::args;
  const std::size_t NR = args("samples", NT2_NB_RANDOM_TEST);
  const double ulpd = args("ulpd",  0.5);

  const T min = args("min", -nt2::Pio_4<T>());
  const T max = args("max", nt2::Pio_4<T>());
  std::cout << "Argument samples #0 chosen in range: [" << min << ",  " << max << "]" << std::endl;
  NT2_CREATE_BUF(a0,T, NR, min, max);

  std::vector<std::pair<T, T> > ref(NR);
  for(std::size_t i=0; i!=NR; ++i)
    ref[i] =  nt2::sincos(a0[i]);

  NT2_COVER_ULP_EQUAL(nt2::tag::fast_sincos_, ((T, a0)), ref, ulpd);
}
