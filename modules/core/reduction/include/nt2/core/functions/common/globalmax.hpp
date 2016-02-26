//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_COMMON_GLOBALMAX_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_COMMON_GLOBALMAX_HPP_INCLUDED

#include <nt2/core/functions/globalmax.hpp>
#include <nt2/include/functions/maximum.hpp>
#include <nt2/include/functions/global.hpp>
#include <nt2/include/functions/is_equal.hpp>
#include <nt2/include/functions/globalfind.hpp>

namespace nt2 { namespace ext
{
  /// INTERNAL ONLY
  BOOST_DISPATCH_IMPLEMENT  ( globalmax_, tag::cpu_,
                              (A0),
                              (unspecified_<A0>)
    )
  {
    typedef typename meta::call<tag::global_(nt2::functor<tag::maximum_>
                                             , const A0&
      )>::type                           result_type;
    BOOST_FORCEINLINE result_type operator()(A0 const& a0) const
    {
      return nt2::global(nt2::functor<tag::maximum_>(), a0);
    }
  };
  /// INTERNAL ONLY
  BOOST_DISPATCH_IMPLEMENT  ( globalmax_, tag::cpu_,
                              (A0)(A1),
                              (unspecified_<A0>)(scalar_<integer_<A1> > )
    )
  {
    typedef typename meta::call<tag::global_(nt2::functor<tag::maximum_>, const A0&)>::type result_type;
    BOOST_FORCEINLINE result_type operator()(A0 const& a0, A1 & a1) const
    {
       result_type tmp =  global(nt2::functor<tag::maximum_>(), a0);
       A1 k = nt2::globalfind(eq(a0, tmp));
       a1 = k;
       return tmp;
    }
  };
} }

#endif
