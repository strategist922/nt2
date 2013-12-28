//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_EXPONENTIAL_FUNCTIONS_SIMD_COMMON_IMPL_LOGS_FAST_F_LOG_HPP_INCLUDED
#define NT2_EXPONENTIAL_FUNCTIONS_SIMD_COMMON_IMPL_LOGS_FAST_F_LOG_HPP_INCLUDED
#include <nt2/include/functions/simd/sqr.hpp>
#include <nt2/include/functions/simd/tofloat.hpp>
#include <nt2/include/functions/simd/is_nan.hpp>
#include <nt2/include/functions/simd/is_ltz.hpp>
#include <nt2/include/functions/simd/is_equal.hpp>
#include <nt2/include/functions/simd/is_eqz.hpp>
#include <nt2/include/functions/simd/is_inf.hpp>
#include <nt2/include/functions/simd/fast_frexp.hpp>
#include <nt2/include/functions/simd/fma.hpp>
#include <nt2/include/functions/simd/multiplies.hpp>
#include <nt2/include/functions/simd/divides.hpp>
#include <nt2/include/functions/simd/seladd.hpp>
#include <nt2/include/functions/simd/if_allbits_else.hpp>
#include <nt2/include/functions/simd/if_else_zero.hpp>
#include <nt2/include/functions/simd/if_else.hpp>
#include <nt2/include/functions/simd/logical_or.hpp>
#include <nt2/include/functions/simd/bitwise_and.hpp>
#include <nt2/include/constants/mone.hpp>
#include <nt2/include/constants/mhalf.hpp>
#include <nt2/include/constants/minf.hpp>
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
#include <nt2/include/functions/rec.hpp>
#include <nt2/include/functions/minusone.hpp>
#include <nt2/sdk/meta/as_logical.hpp>
#include <nt2/include/functions/splat.hpp>

namespace nt2 { namespace details
{
  //////////////////////////////////////////////////////////////////////////////
  // how to compute the various logaithm
  // first reduce the the data
  // a0 is supposed > 0
  // the input a0 is split into a mantissa and an exponent
  // the mantissa m is between sqrt(0.5) and sqrt(2) and the correspondint exponent is e
  // a0 = m*2^e
  // then the log? calculus is split in two parts (? being nothing: natural logarithm,  2: base 2 logarithm,  10 base ten logarithm)
  // as log?(a) = log?(2^e)+log?(m)
  // 1) computing log?(m)
  //   first put x = m-1 (so -0.29 <  x < 0.414)
  //   write log(m)   = log(1+x)   = x + x*x/2 + x*x*x*g(x)
  //   write log2(m)  = log2(1+x)  = C2*log(x)   C2 =  log(2)  the multiplication have to be taken seriously as C2 is not exact
  //   write log10(m) = log10(1+x) = C10*log(x)  C10=  log(10) the multiplication have to be taken seriously as C10 is not exact
  // then g(x) has to be approximated
  // g is ((log(1+x)/x-1)/x-1/2)/x
  // It is not a good idea to approximate directly log(1+x) instead of g,  because this will lead to bad precision around 1.
  //
  // in this approximation one can choose a best approximation rational function given by remez algorithm.
  // there exist a classical solution which is a polynomial p8 one of degree 8 that gives 0.5ulps everywhere
  // the classical compution being to consider p8 = xp3_1(x*x)+p4_2(x*x) this can be a gain of 2 cycles on the pipeline
  // this is what is done in the log impl;
  // Now,  it is possible to choose a rational fraction or a polynomial of lesser degree to approximate g
  // providing faster but less accurate logs.
  // 2) computing log?(2^e)
  // see the explanations relative to each case
  // 3) finalize
  // This is simply treating invalid entries
  //////////////////////////////////////////////////////////////////////////////

  template < class A0 >
  struct fast_logarithm< A0, tag::simd_type, float>
  {
    static inline void kernel_log(const A0& a0,
                                  A0& fe,
                                  A0& x,
                                  A0& x2,
                                  A0& y)
    {
      typedef typename meta::as_integer<A0, signed>::type int_type;
      typedef typename meta::as_logical<A0>::type              lA0;
      int_type e;
      nt2::fast_frexp(a0, x, e);
      lA0 xltsqrthf = lt(x, Sqrt_2o_2<A0>()); //single_constant<A0, 0x3f3504f3>());
      fe = seladd(xltsqrthf, nt2::tofloat(e), Mone<A0>());
      x =  minusone(seladd(xltsqrthf, x, x));


      typedef typename meta::scalar_of<A0>::type sA0;
      x2 = sqr(x);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// These are different choices of polynomials p and q such that g(x) is approximated by p/q
// The evaluations have 3 modularities depending if p or q is a constant or of the form q+x or complete polynomials
// 1)
//   static const boost::array<sA0, 9 > p= {{   7.0376836e-02,  -1.1514610e-01,   1.1676998e-01,  -1.2420141e-01,
//                                              1.4249323e-01,  -1.6668057e-01,   2.0000714e-01,  -2.4999994e-01,
//                                              3.3333331e-01 }};                                                  //9.2  cycles/value // 0.5 ulp max
// q = 1
//              evalution by polevl(p, x)
// 2)
// p= 1                                                                                                            // 6.4 cycles/value  // 4  ulp max
// static const boost::array<sA0, 4 > q= {{6.4806163e-02,  -1.1776499e-01,   2.2500458e+00,   3.0000787e+00}};
//              evalution by rec(polevl(q, x));
// 3)
//  p= 1                                                                                                           // 5.8 cycles/value  // 32 ulp max
//  static const boost::array<sA0, 3 > q= {{-1.1320251e-01,   2.2559700e+00,   3.0001562e+00}};
//              evalution by rec(polevl(q, x));
// 4)
//  static const boost::array<sA0, 2 > p= {{   8.2778692e-02,   1.4085096e-01}};                                   //5.6  cycles/value // 1.0 ulp max
//  static const boost::array<sA0, 3 > q= {{   1.7031030e-01,   5.6524682e-01,   4.2255422e-01}};
//              evalution by polevl(p, x)/(polevl(q, x));
// 5)
//  static const boost::array<sA0, 2 > p= {{ 8.2236594e-03,   1.6316444e-01 }};                                    // 5.8 cycles/value  // 32 ulp max
//  static const boost::array<sA0, 2 > q= {{ 3.9268133e-01,  4.8951855e-01 }};
//              evalution by polevl(p, x)/(polevl(q, x));
// 6)
//  static const boost::array<sA0, 3 > p= {{ -6.0903007e-04,  -8.6939648e-02,  -1.2743619e-01 }};                  // 6.3 cycles/value  // 0.5 ulp max
//  static const boost::array<sA0, 3 > q= {{ -1.8310602e-01,  -5.4754895e-01,  -3.8230851e-01 }};
// 7)
// static const boost::array<sA0, 4 > p= {{ -1.5532331e-01,   2.1585755e-01,  -2.5109935e-01,   3.3309942e-01 }};  // 5.7  cycles/value // 512ulp max
// q = 1
//  evalution by polevl(p, x)
// 8)
      static const boost::array<sA0, 3 > p= {{   -9.8859705e-03,   3.3903550e-02,   3.9923689e-01 }};                // 5.2 cycles/value  // 8  ulp max
      static const A0 q = splat<A0>(  1.1977361e+00 ); // the denom is q+x
      y =  polevl(x, p)/(q+x);


// static const boost::array<sA0, 3 > p= {{1.9163288e-01,  -2.6443884e-01,   3.3371061e-01}};                      // 5.1  cycles/value // 512ulp max
// q = 1
//  evalution by rec(polevl(q, x))
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      y*= x*x2;
    }

    static inline A0 log(const A0& a0)
    {
      //log(2.0) in double is 6.931471805599453e-01
      //double(0.693359375f)+double(-0.00021219444f)  is  6.931471805600000e-01 at 1.0e-14 of log(2.0)
      // let us call Log_2hi 0.693359375f anf Log_2lo -0.00021219444f
      // We use thi to correct the sum where this could matter a lot
      // log(a0) = fe*Log_2hi+ (0.5f*x*x +(fe*Log_2lo+y))
      // These operations are order dependant: the parentheses do matter
      A0 x, fe, x2, y;
      kernel_log(a0, fe, x, x2, y);
      y = nt2::fma(fe, Log_2lo<A0>(), y);
      y = nt2::fma(Mhalf<A0>(), x2, y);
      A0 z  = x + y;
      return  nt2::fma(Log_2hi<A0>(), fe, z);
    }

    static inline A0 log2(const A0& a0)
    {
      //here let l2em1 = log2(e)-1, the computation is done as:
      //log2(a0) = ((l2em1*x+(l2em1*(y+x*x/2)))+(y+x*x/2)))+x+fe for best results
      // once again the order is very important.
      A0 x, fe, x2, y;
      kernel_log(a0, fe, x, x2, y);
      y =  nt2::fma(Mhalf<A0>(),x2, y);
      // multiply log of fraction by log2(e)
      A0 z = nt2::fma(x,Log2_em1<A0>(),y*Log2_em1<A0>());// 0.44269504088896340735992
      return ((z+y)+x)+fe;
    }

    static inline A0 log10(const A0& a0)
    {
      // here there are two multiplication:  log of fraction by log10(e) and base 2 exponent by log10(2)
      // and we have to split log10(e) and log10(2) in two parts to get extra precision when needed
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
