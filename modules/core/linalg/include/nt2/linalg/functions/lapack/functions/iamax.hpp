//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_IAMAX_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_IAMAX_HPP_INCLUDED

#include <nt2/linalg/functions/iamax.hpp>

#include <nt2/linalg/details/utility/f77_wrapper.hpp>

#include <nt2/include/functions/height.hpp>

extern "C"
{
  long int NT2_F77NAME(idamax)(const nt2_la_int* n, const double *dx
                              ,const nt2_la_int* incx);

  long int NT2_F77NAME(isamax)(const nt2_la_int* n, const float *dx
                              ,const nt2_la_int* incx);

  long int NT2_F77NAME(icamax)(const nt2_la_int* n, const nt2_la_complex *dx
                              ,const nt2_la_int* incx);

  long int NT2_F77NAME(izamax)(const nt2_la_int* n, const nt2_la_complex *dx
                              ,const nt2_la_int* incx);
}

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::iamax_, tag::cpu_
                            , (A0)(S0)
                            , ((container_< nt2::tag::table_, double_<A0>, S0 >))
                            )
  {
     typedef long int result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0) const
     {
        nt2_la_int  n  = nt2::height(a0);
        nt2_la_int  incx = 1;

        return NT2_F77NAME(idamax)(&n, a0.raw(), &incx);
     }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::iamax_, tag::cpu_
                            , (A0)(S0)
                            , ((container_< nt2::tag::table_, single_<A0>, S0 >))
                            )
  {
     typedef long int result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0) const
     {
        nt2_la_int  n  = nt2::height(a0);
        nt2_la_int  incx = 1;

        return NT2_F77NAME(isamax)(&n, a0.raw(), &incx);
     }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::iamax_, tag::cpu_
                            , (A0)(S0)
                            , ((container_< nt2::tag::table_, complex_<single_<A0> > , S0 >))
                            )
  {
     typedef long int result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0) const
     {
        nt2_la_int  n  = nt2::height(a0);
        nt2_la_int  incx = 1;

        return NT2_F77NAME(icamax)(&n, a0.raw(), &incx);
     }

  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::iamax_, tag::cpu_
                            , (A0)(S0)
                            , ((container_< nt2::tag::table_, complex_< double_<A0> >, S0 > ))
                            )
  {
     typedef long int result_type;

     BOOST_FORCEINLINE result_type operator()(A0& a0) const
     {
        nt2_la_int  n  = nt2::height(a0);
        nt2_la_int  incx = 1;

        return NT2_F77NAME(izamax)(&n, a0.raw(), &incx);
     }

  };

} }

#endif
