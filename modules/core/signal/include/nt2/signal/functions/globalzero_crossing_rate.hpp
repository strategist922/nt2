//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_GLOBALZERO_CROSSING_RATE_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_GLOBALZERO_CROSSING_RATE_HPP_INCLUDED
#include <nt2/include/functor.hpp>
#include <nt2/include/functions/zero_crossing_rate.hpp>
#include <nt2/include/functions/colvect.hpp>

namespace nt2
{
  /*!
    @brief rate of sign changes along a signal

    Computes the rate of sign changes along all elements of a signal

    @par Semantic

    For any table expression and integer:

    @code
    auto r = globalzero_crossing_rate(s, n);
    @endcode

    is equivalent to:

    @code
    auto r = zero_crossing_rate(s(_))(1);
    @endcode

    n default to firstnonsingleton(s)

    @see @funcref{zero_crossing_rate}, @funcref{colon},

    @param a0 Table expression to process

    @return An expression eventually evaluated to the result
  **/

  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalzero_crossing_rate(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( nt2::zero_crossing_rate(nt2::colvect(a0)));

}
#endif
