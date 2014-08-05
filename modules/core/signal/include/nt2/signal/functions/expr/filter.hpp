//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_EXPR_FILTER_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_EXPR_FILTER_HPP_INCLUDED

#include <nt2/signal/functions/filter.hpp>
#include <nt2/signal/details/as_filter.hpp>

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::filter_, boost::simd::tag::simd_
                            , (A0)(A1)(A2)
                            , ((ast_<A0, nt2::container::domain>))
                              (unspecified_<A1>)
                              (unspecified_<A2>)
                            )
  {
    BOOST_DISPATCH_RETURNS( 3
                          , (A0 const& a0,A1 const& a1,A2 const& a2)
                          , nt2::filter(nt2::as_filter(a0),a1,a2)
                          );
  };
} }

#endif
