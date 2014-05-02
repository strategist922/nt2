//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_GER_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_GER_HPP_INCLUDED

#include <nt2/linalg/functions/ger.hpp>
#include <nt2/linalg/details/utility/f77_wrapper.hpp>

#include <nt2/include/functions/height.hpp>

extern "C"
{
  void NT2_F77NAME(dger)(const nt2_la_int* M, const nt2_la_int* N, const double* alpha,
                        const double* dx, const nt2_la_int* incx, const double* dy,
                        const nt2_la_int* incy, double* A, const nt2_la_int* lda);

  void NT2_F77NAME(sger)(const nt2_la_int* M, const nt2_la_int* N, const float* alpha,
                         const float* dx, const nt2_la_int* incx, const float* dy,
                         const nt2_la_int* incy, float* A, const nt2_la_int* lda);

  long int NT2_F77NAME(cgeru)(const nt2_la_int* M, const nt2_la_int* N, const nt2_la_complex* alpha,
                              const nt2_la_complex* dx, const nt2_la_int* incx, const nt2_la_complex* dy,
                              const nt2_la_int* incy, nt2_la_complex* A, const nt2_la_int* lda);

  long int NT2_F77NAME(zgeru)(const nt2_la_int* M, const nt2_la_int* N, const nt2_la_complex* alpha,
                              const nt2_la_complex* dx, const nt2_la_int* incx, const nt2_la_complex* dy,
                              const nt2_la_int* incy, nt2_la_complex* A, const nt2_la_int* lda);
}

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ger_, tag::cpu_
                            , ((container_< nt2::tag::table_, double_<A0>, S0 >))
                            , ((container_< nt2::tag::table_, double_<A1>, S1 >))
                            , ((container_< nt2::tag::table_, double_<A2>, S2 >))
                            , (scalar_< double_<A3> >)
                            )
  {
     typedef void result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0, A1& a1, A2& a2, const A3 & a3) const
     {
        nt2_la_int  m    = nt2::height(a2);
        nt2_la_int  n    = nt2::width(a2);
        nt2_la_int  lda  = a2.leading.size();
        double alpha   = a3;
        nt2_la_int  incx = 1;
        nt2_la_int  incy = a1.leading_size();

        NT2_F77NAME(dger)(&m, &n, &alpha, a0.raw(), &incx, a1.raw(), &incy, a3.raw(), &lda);
     }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ger_, tag::cpu_
                            , ((container_< nt2::tag::table_, float_<A0>, S0 >))
                            , ((container_< nt2::tag::table_, float_<A1>, S1 >))
                            , ((container_< nt2::tag::table_, float_<A2>, S2 >))
                            , (scalar_< float_<A3> >)
                            )
  {
     typedef void result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0, A1& a1, A2& a2, const A3 & a3) const
     {
        nt2_la_int  m    = nt2::height(a2);
        nt2_la_int  n    = nt2::width(a2);
        nt2_la_int  lda  = a2.leading.size();
        float alpha      = a3;
        nt2_la_int  incx = 1;
        nt2_la_int  incy = a1.leading_size();

        NT2_F77NAME(sger)(&m, &n, &alpha, a0.raw(), &incx, a1.raw(), &incy, a3.raw(), &lda);
     }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ger_, tag::cpu_
                            , ((container_< nt2::tag::table_, float_<A0>, S0 >))
                            , ((container_< nt2::tag::table_, float_<A1>, S1 >))
                            , ((container_< nt2::tag::table_, float_<A2>, S2 >))
                            , (scalar_< complex_< float_<A3> > >)
                            )
  {
     typedef void result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0, A1& a1, A2& a2, const A3 & a3) const
     {
        nt2_la_int  m    = nt2::height(a2);
        nt2_la_int  n    = nt2::width(a2);
        nt2_la_int  lda  = a2.leading.size();
        nt2_la_complex alpha = a3;
        nt2_la_int  incx = 1;
        nt2_la_int  incy = a1.leading_size();

        NT2_F77NAME(cgeru)(&m, &n, &alpha, a0.raw(), &incx, a1.raw(), &incy, a3.raw(), &lda);
     }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ger_, tag::cpu_
                            , ((container_< nt2::tag::table_, complex_< double_<A0> >, S0 > ))
                            , ((container_< nt2::tag::table_, complex_< double_<A1> >, S1 > ))
                            , ((container_< nt2::tag::table_, complex_< double_<A2> >, S2 > ))
                            , (scalar_< complex_< double_<A3> > >)
                            )
  {
     typedef void result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0, A1& a1, A2& a2, const A3 & a3) const
     {
        nt2_la_int  m    = nt2::height(a2);
        nt2_la_int  n    = nt2::width(a2);
        nt2_la_int  lda  = a2.leading.size();
        nt2_la_complex alpha = a3;
        nt2_la_int  incx = 1;
        nt2_la_int  incy = a1.leading_size();

        NT2_F77NAME(zgeru)(&m, &n, &alpha, a0.raw(), &incx, a1.raw(), &incy, a3.raw(), &lda);
     }
  };
} }

#endif
