//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_PREDICATES_FUNCTIONS_TABLE_ISINSIDE_HPP_INCLUDED
#define NT2_PREDICATES_FUNCTIONS_TABLE_ISINSIDE_HPP_INCLUDED

#include <nt2/predicates/functions/isinside.hpp>
#include <boost/fusion/adapted/mpl.hpp>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( isinside_, tag::cpu_
                            , (A0)(A1)(N)
                            , ((fusion_sequence_<A0,N>))
                              ((ast_<A1, nt2::container::domain>))
                            )
  {
    BOOST_DISPATCH_RETURNS( 2, (A0 const& a0, A1 const& a1)
                          , isinside(a0,a1.extent())
                          )
  };
} }

#endif
