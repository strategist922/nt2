//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/simd/include/functions/scalar/fast_divides.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/timing.hpp>

typedef NT2_SIMD_DEFAULT_EXTENSION  ext_t;
using boost::simd::tag::fast_divides_;

#define RS(T,V1,V2) (T, (V1) ,(V2))

namespace n1
{
  typedef float T;
  typedef boost::simd::native<T,ext_t> vT;
  NT2_TIMING(fast_divides_,( RS(vT,T(-100),T(100)) )( RS(vT,T(-100),T(100)) ) )
}
namespace n2
{
  typedef double T;
  typedef boost::simd::native<T,ext_t> vT;
  NT2_TIMING(fast_divides_,( RS(vT,T(-100),T(100)) )( RS(vT,T(-100),T(100)) ) )
}

#undef RS
