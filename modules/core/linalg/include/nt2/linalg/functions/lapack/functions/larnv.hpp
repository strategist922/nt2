//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_LARNV_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_LAPACK_FUNCTIONS_LARNV_HPP_INCLUDED

#include <nt2/linalg/functions/larnv.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/linalg/details/utility/f77_wrapper.hpp>

extern "C"
{
  void NT2_F77NAME(dlarnv)( const nt2_la_int* idist , nt2_la_int* iseed
                          , const nt2_la_int* n     , double* x
                          );

  void NT2_F77NAME(slarnv)( const nt2_la_int* idist , nt2_la_int* iseed
                          , const nt2_la_int* n     , float* x
                          );

  void NT2_F77NAME(zlarnv)( const nt2_la_int* idist , nt2_la_int* iseed
                          , const nt2_la_int* n     , nt2_la_complex* x
                          );

  void NT2_F77NAME(clarnv)( const nt2_la_int* idist , nt2_la_int* iseed
                          , const nt2_la_int* n     , nt2_la_complex* x
                          );
}

namespace nt2 { namespace ext
{
  /// INTERNAL ONLY
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::larnv_, tag::cpu_
                            , (A0)(A1)(S1)(A2)(S2)
                            , (scalar_< integer_<A0> >)
                              ((container_< nt2::tag::table_, integer_<A1>, S1 >))
                              ((container_< nt2::tag::table_, double_<A2>, S2 >))
                            )
  {
    typedef void  result_type;

    BOOST_FORCEINLINE result_type operator()(A0 const & a0, A1& a1, A2& a2) const
    {
      nt2_la_int idist = a0;
      nt2_la_int n = nt2::numel(a2);

      NT2_F77NAME(dlarnv) ( &idist, a1.raw(), &n, a2.raw() );
    }
  };

  /// INTERNAL ONLY
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::larnv_, tag::cpu_
                            , (A0)(A1)(S1)(A2)(S2)
                            , (scalar_< integer_<A0> >)
                              ((container_< nt2::tag::table_, integer_<A1>, S1 >))
                              ((container_< nt2::tag::table_, single_<A2>, S2 >))
                            )
  {
    typedef void  result_type;

    BOOST_FORCEINLINE result_type operator()(A0 const & a0, A1& a1, A2& a2) const
    {
      nt2_la_int idist = a0;
      nt2_la_int n = nt2::numel(a2);

      NT2_F77NAME(slarnv) ( &idist, a1.raw(), &n, a2.raw() );
    }
  };

  /// INTERNAL ONLY
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::larnv_, tag::cpu_
                            , (A0)(A1)(S1)(A2)(S2)
                            , (scalar_< integer_<A0> >)
                              ((container_< nt2::tag::table_, integer_<A1>, S1 >))
                              ((container_< nt2::tag::table_, complex_<single_<A2> > , S2 >))
                            )
  {
    typedef void  result_type;

    BOOST_FORCEINLINE result_type operator()(A0 const & a0, A1& a1, A2& a2) const
    {
      nt2_la_int idist = a0;
      nt2_la_int n = nt2::numel(a2);

      NT2_F77NAME(clarnv) ( &idist, a1.raw(), &n, a2.raw() );
    }
  };

  /// INTERNAL ONLY
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::larnv_, tag::cpu_
                            , (A0)(A1)(S1)(A2)(S2)
                            , (scalar_< integer_<A0> >)
                              ((container_< nt2::tag::table_, integer_<A1>, S1 >))
                              ((container_< nt2::tag::table_, complex_<double_<A2> > , S2 >))
                            )
  {
    typedef void  result_type;

    BOOST_FORCEINLINE result_type operator()(A0 const & a0, A1& a1, A2& a2) const
    {
      nt2_la_int idist = a0;
      nt2_la_int n = nt2::numel(a2);

      NT2_F77NAME(zlarnv) ( &idist, a1.raw(), &n, a2.raw() );
    }
  };

} }

#endif
