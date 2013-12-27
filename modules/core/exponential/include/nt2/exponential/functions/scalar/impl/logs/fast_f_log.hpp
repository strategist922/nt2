//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_EXPONENTIAL_FUNCTIONS_SCALAR_IMPL_LOGS_FAST_F_LOG_HPP_INCLUDED
#define NT2_EXPONENTIAL_FUNCTIONS_SCALAR_IMPL_LOGS_FAST_F_LOG_HPP_INCLUDED

#include <nt2/include/functions/scalar/sqr.hpp>
#include <nt2/include/functions/scalar/bitwise_and.hpp>
#include <nt2/include/functions/scalar/tofloat.hpp>
#include <nt2/include/functions/scalar/is_nan.hpp>
#include <nt2/include/functions/scalar/is_ltz.hpp>
#include <nt2/include/functions/scalar/is_eqz.hpp>
#include <nt2/include/functions/scalar/fast_frexp.hpp>
#include <nt2/include/functions/scalar/genmask.hpp>
#include <nt2/include/functions/scalar/fma.hpp>
#include <nt2/include/functions/scalar/multiplies.hpp>
#include <nt2/include/functions/scalar/unary_minus.hpp>
#include <nt2/include/functions/scalar/plus.hpp>
#include <nt2/include/constants/nan.hpp>
#include <nt2/include/constants/inf.hpp>
#include <nt2/include/constants/minf.hpp>
#include <nt2/include/constants/mhalf.hpp>
#include <nt2/include/constants/zero.hpp>
#include <nt2/include/constants/log_2hi.hpp>
#include <nt2/include/constants/log_2lo.hpp>
#include <nt2/include/constants/log2_em1.hpp>
#include <nt2/include/constants/log10_ehi.hpp>
#include <nt2/include/constants/log10_elo.hpp>
#include <nt2/include/constants/log10_2hi.hpp>
#include <nt2/include/constants/log10_2lo.hpp>
#include <nt2/include/constants/sqrt_2o_2.hpp>
#include <boost/simd/sdk/config.hpp>
#include <nt2/include/functions/polevl.hpp>
#include <boost/array.hpp>
#include <nt2/include/constants/log_2.hpp>

namespace nt2 { namespace details
{
  template < class A0,
             class Style ,
             class base_A0 = typename meta::scalar_of<A0>::type>
             struct fast_logarithm{};

  //////////////////////////////////////////////////////////////////////////////
  // math log functions
  //////////////////////////////////////////////////////////////////////////////

  template < class A0 >
  struct fast_logarithm< A0, tag::not_simd_type, float>
  {

    static inline void kernel_log(const A0& a0,
                                  A0& fe,
                                  A0& x,
                                  A0& x2,
                                  A0& y)
    {
      typedef typename meta::as_integer<A0, signed>::type int_type;
      int_type e = 0;
      nt2::fast_frexp(a0, x, e);
      int_type x_lt_sqrthf = -(x < Sqrt_2o_2<A0>());
      e += x_lt_sqrthf;
      x += nt2::bitwise_and(x, x_lt_sqrthf)+Mone<A0>();
      static const boost::array<A0, 3 > p= {{   -9.8859705e-03,   3.3903550e-02,   3.9923689e-01 }};                // 5.2 cycles/value  // 4  ulp max
      const A0 q = 1.1977361e+00;
      y =  polevl(x, p)/(q+x);
      x2 = sqr(x);
      y*= x*x2;
      fe = nt2::tofloat(e);
    }

    static inline A0 log(const A0& a0)
    {
      if (a0 == nt2::Inf<A0>()) return a0;
      if (nt2::is_eqz(a0)) return nt2::Minf<A0>();
#ifdef BOOST_SIMD_NO_NANS
      if (nt2::is_ltz(a0)) return nt2::Nan<A0>();
#else
      if (nt2::is_nan(a0)||nt2::is_ltz(a0)) return nt2::Nan<A0>();
#endif

      A0 x, fe, x2, y;
      kernel_log(a0, fe, x, x2, y);
      y = nt2::fma(fe, Log_2lo<A0>(), y);
      y = nt2::fma(Mhalf<A0>(), x2, y);
      A0 z  = x + y;
      return nt2::fma(Log_2hi<A0>(), fe, z);
    }

    static inline A0 log2(const A0& a0)
    {
      if (a0 == nt2::Inf<A0>()) return a0;
      if (nt2::is_eqz(a0)) return nt2::Minf<A0>();
#ifdef BOOST_SIMD_NO_NANS
      if (nt2::is_ltz(a0)) return nt2::Nan<A0>();
#else
      if (nt2::is_nan(a0)||nt2::is_ltz(a0)) return nt2::Nan<A0>();
#endif
      A0 x, fe, x2, y;
      kernel_log(a0, fe, x, x2, y);
      y =  nt2::fma(Mhalf<A0>(),x2, y);
      // multiply log of fraction by log2(e)
      A0 z = nt2::fma(x,Log2_em1<A0>(),y*Log2_em1<A0>());// 0.44269504088896340735992
      return ((z+y)+x)+fe;
    }

    static inline A0 log10(const A0& a0)
    {
      if (a0 == nt2::Inf<A0>()) return a0;
      if (nt2::is_eqz(a0)) return nt2::Minf<A0>();
#ifdef BOOST_SIMD_NO_NANS
      if (nt2::is_ltz(a0)) return nt2::Nan<A0>();
#else
      if (nt2::is_nan(a0)||nt2::is_ltz(a0)) return nt2::Nan<A0>();
#endif
      A0 x, fe, x2, y;
      kernel_log(a0, fe, x, x2, y);
      y =  nt2::amul(y, Mhalf<A0>(), x2);
      A0 z = mul(x+y, Log10_elo<A0>());
      z = nt2::amul(z, y, Log10_ehi<A0>());
      z = nt2::amul(z, x, Log10_ehi<A0>());
      z = nt2::amul(z, fe, Log10_2hi<A0>());
      return nt2::amul(z, fe, Log10_2lo<A0>());
    }
  };
} }


#endif
